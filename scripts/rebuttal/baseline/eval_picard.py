#!/usr/bin/env python3
"""
PICARD-style source-constrained decoding (adapted from Scholak et al., 2021).

Once the last `trigger_length` output tokens match the input context, decoding enters COPY
mode: for up to `copy_budget` steps, greedy argmax is restricted to tokens that continue a
matching context span. After COPY ends, `cooldown_tokens` free steps follow and COPY stays
disarmed until the output suffix no longer matches the context.

--reference-block-acceleration verifies LLMA-style context drafts during free steps without
changing the decoding policy. If any block row has a top-2 logit margin <= 0.5, the whole
row is decoded again without blocks, so BF16 near-ties are resolved as in serial decoding.

Example:
  python scripts/rebuttal/baseline/eval_picard.py \
    --model-path MODEL_PATH \
    --oracle data/contract/test.jsonl \
    --output outputs/rebuttal/baselines/qwen3_8b/contract/picard.jsonl \
    --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv
from model.prompt_utils import encode_editor_prefix

apply_cuda_visible_devices_from_argv(sys.argv)

import torch
from tqdm.auto import tqdm

from scripts.rebuttal.baseline.eval_copyspec import forward, load_model, load_rows, speed_summary
from scripts.rebuttal.baseline.eval_llma import build_reference_index, longest_reference_match

BLOCK_MIN_MATCH_LENGTH = 1
BLOCK_MAX_MATCH_LENGTH = 5
BLOCK_SIZE = 20
BLOCK_MIN_LOGIT_MARGIN = 0.5
BLOCK_STAT_KEYS = (
    "reference_block_attempts",
    "reference_block_proposed_tokens",
    "reference_block_accepted_tokens",
    "reference_block_low_margin_fallbacks",
)


def build_source_index(reference: list[int], trigger_length: int) -> dict:
    index: dict[tuple[int, ...], list[int]] = {}
    for start in range(len(reference) - trigger_length + 1):
        index.setdefault(tuple(reference[start : start + trigger_length]), []).append(start)
    return index


def grounded_states(index: dict, generated: list[int], reference: list[int], trigger_length: int) -> list[int]:
    """Context positions right after each occurrence of the last `trigger_length` output tokens."""
    starts = index.get(tuple(generated[-trigger_length:]), ())
    return [start + trigger_length for start in starts if start + trigger_length < len(reference)]


def transitions(reference: list[int], states: list[int]) -> dict[int, list[int]]:
    """Admissible next tokens, each mapped to the context positions it leads to."""
    result: dict[int, list[int]] = {}
    for position in states:
        if position < len(reference):
            result.setdefault(reference[position], []).append(position + 1)
    return result


def constrained_argmax(logits: torch.Tensor, token_ids) -> int:
    token_ids = sorted(token_ids)
    index = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    return token_ids[int(logits.index_select(0, index).argmax())]


def singleton_run(reference: list[int], states: list[int], budget: int, max_tokens: int, eos_token_id: int) -> list[int]:
    """COPY steps whose admissible set is a single token; they do not depend on the logits."""
    run = []
    while states and budget > 0 and len(run) < max_tokens:
        options = transitions(reference, states)
        if len(options) != 1:
            break
        token, states = next(iter(options.items()))
        run.append(token)
        budget -= 1
        if token == eos_token_id:
            break
    return run


class SourceConstraint:
    """COPY/GEN state of the source-span constraint."""

    def __init__(self, reference, index, trigger_length, copy_budget, cooldown_tokens, eos_token_id):
        self.reference = reference
        self.index = index
        self.trigger_length = trigger_length
        self.copy_budget = copy_budget
        self.cooldown_tokens = cooldown_tokens
        self.eos_token_id = eos_token_id
        self.states: list[int] = []  # live context positions; non-empty means COPY mode
        self.budget = 0
        self.cooldown = 0
        self.disarmed = False
        self.stats = {
            "trigger_count": 0,
            "constrained_steps": 0,
            "forced_override_tokens": 0,
            "budget_exits": 0,
            "source_exhausted_exits": 0,
            "eos_exits": 0,
            "rearm_count": 0,
        }

    def try_trigger(self, generated: list[int]) -> bool:
        states = grounded_states(self.index, generated, self.reference, self.trigger_length)
        if self.disarmed:
            if not states:
                self.disarmed = False
                self.stats["rearm_count"] += 1
            return False
        if states:
            self.states = states
            self.budget = self.copy_budget
            self.stats["trigger_count"] += 1
            return True
        return False

    def options(self) -> dict[int, list[int]]:
        return transitions(self.reference, self.states)

    def copy(self, token: int, greedy_token: int) -> None:
        self.stats["constrained_steps"] += 1
        self.stats["forced_override_tokens"] += int(token != greedy_token)
        self.states = self.options()[token]
        self.budget -= 1
        if token == self.eos_token_id:
            self.stats["eos_exits"] += 1
            self.states = []
        elif not self.options():
            self.stats["source_exhausted_exits"] += 1
            self.exit_copy()
        elif self.budget == 0:
            self.stats["budget_exits"] += 1
            self.exit_copy()

    def exit_copy(self) -> None:
        self.states = []
        self.cooldown = self.cooldown_tokens
        self.disarmed = True

    def free(self) -> None:
        if self.cooldown:
            self.cooldown -= 1


@torch.no_grad()
def picard_decode(
    model,
    prefix_ids: torch.Tensor,
    reference: list[int],
    source_index: dict,
    block_index: dict | None,
    *,
    eos_token_id: int,
    max_new_tokens: int,
    trigger_length: int,
    copy_budget: int,
    cooldown_tokens: int,
) -> tuple[list[int], dict]:
    device = prefix_ids.device
    prompt_length = prefix_ids.shape[1]
    constraint = SourceConstraint(reference, source_index, trigger_length, copy_budget, cooldown_tokens, eos_token_id)
    block_stats = dict.fromkeys(BLOCK_STAT_KEYS, 0)

    outputs = forward(model, prefix_ids)
    past_key_values = outputs.past_key_values
    generated = [int(outputs.logits[0, -1].argmax())]

    while len(generated) < max_new_tokens and generated[-1] != eos_token_id:
        if not constraint.states and constraint.cooldown == 0:
            constraint.try_trigger(generated)

        if constraint.states:
            run = singleton_run(
                reference, constraint.states, constraint.budget, max_new_tokens - len(generated), eos_token_id
            )
            if run:
                # forced tokens are known in advance, so one forward scores all of them
                block = torch.tensor([[generated[-1], *run[:-1]]], dtype=torch.long, device=device)
                outputs = forward(model, block, past_key_values, all_logits=True)
                past_key_values = outputs.past_key_values
                for token, greedy_token in zip(run, outputs.logits[0].argmax(dim=-1).tolist()):
                    constraint.copy(token, greedy_token)
                    generated.append(token)
                continue

        if block_index is not None and not constraint.states:
            match = longest_reference_match(block_index, generated, BLOCK_MIN_MATCH_LENGTH, BLOCK_MAX_MATCH_LENGTH)
            draft = []
            if match is not None:
                start, length = match
                remaining = max_new_tokens - len(generated)
                draft = reference[start + length : start + length + min(BLOCK_SIZE - 1, remaining - 1)]
            if draft:
                block_stats["reference_block_attempts"] += 1
                block_stats["reference_block_proposed_tokens"] += len(draft)
                block = torch.tensor([[generated[-1], *draft]], dtype=torch.long, device=device)
                outputs = forward(model, block, past_key_values, all_logits=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0]
                greedy = logits.argmax(dim=-1).tolist()
                low_margin = False
                stopped = False
                for offset, token in enumerate(draft):
                    top2 = torch.topk(logits[offset].float(), k=2).values
                    if float(top2[0] - top2[1]) <= BLOCK_MIN_LOGIT_MARGIN:
                        block_stats["reference_block_low_margin_fallbacks"] += 1
                        low_margin = True
                        break
                    # the emitted token is always the greedy one; a mismatch ends the block
                    generated.append(greedy[offset])
                    constraint.free()
                    if token != greedy[offset]:
                        stopped = True
                        break
                    block_stats["reference_block_accepted_tokens"] += 1
                    # also stop before a step at which the serial policy would enter COPY
                    if token == eos_token_id or (constraint.cooldown == 0 and constraint.try_trigger(generated)):
                        stopped = True
                        break
                if not low_margin and not stopped and len(generated) < max_new_tokens:
                    generated.append(greedy[len(draft)])
                    constraint.free()
                past_key_values.crop(prompt_length + len(generated) - 1)
                if not low_margin:
                    continue

        block = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        outputs = forward(model, block, past_key_values)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1]
        greedy_token = int(logits.argmax())
        if constraint.states:
            token = constrained_argmax(logits, constraint.options())
            constraint.copy(token, greedy_token)
        else:
            token = greedy_token
            constraint.free()
        generated.append(token)

    stats = constraint.stats
    if block_index is not None:
        stats.update(block_stats)
    return generated[:max_new_tokens], stats


def main() -> None:
    parser = argparse.ArgumentParser(description="PICARD-style source-constrained greedy decoding.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--trigger-length", type=int, default=5)
    parser.add_argument("--copy-budget", type=int, default=16)
    parser.add_argument("--cooldown-tokens", type=int, default=1)
    parser.add_argument("--reference-block-acceleration", action="store_true")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    rows = load_rows(Path(args.oracle), limit=args.max_samples)
    tokenizer, model = load_model(args.model_path)
    device = torch.device("cuda")
    eos_token_id = tokenizer.eos_token_id
    print(
        f"loaded {args.model_path} trigger={args.trigger_length} budget={args.copy_budget} "
        f"cooldown={args.cooldown_tokens} reference_blocks={args.reference_block_acceleration} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    settings = dict(
        eos_token_id=eos_token_id,
        max_new_tokens=args.max_new_tokens,
        trigger_length=args.trigger_length,
        copy_budget=args.copy_budget,
        cooldown_tokens=args.cooldown_tokens,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_rows: list[list[int]] = []
    totals: dict[str, int] = {}
    total_seconds = 0.0
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="picard", unit="sample"):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            prefix_ids = encode_editor_prefix(
                tokenizer, row["context"], row["instruction"], device, disable_thinking=args.disable_thinking
            )
            reference = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
            source_index = build_source_index(reference, args.trigger_length)
            block_index = None
            if args.reference_block_acceleration:
                block_index = build_reference_index(reference, BLOCK_MIN_MATCH_LENGTH, BLOCK_MAX_MATCH_LENGTH)
            token_ids, stats = picard_decode(model, prefix_ids, reference, source_index, block_index, **settings)
            if block_index is not None:
                stats["reference_block_row_fallback"] = int(stats["reference_block_low_margin_fallbacks"] > 0)
                if stats["reference_block_row_fallback"]:
                    token_ids, serial_stats = picard_decode(model, prefix_ids, reference, source_index, None, **settings)
                    stats.update(serial_stats)
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start_time

            total_seconds += seconds
            token_rows.append(token_ids)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            out_row = dict(row)
            out_row["prediction"] = prediction
            out_row["model_edited_answer"] = prediction
            out_row["model_name"] = args.model_path
            out_row["generated_token_ids"] = token_ids
            out_row["picard_stats"] = stats
            out_row["generation_seconds"] = seconds
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "model_path": args.model_path,
        "oracle": args.oracle,
        "sample_count": len(rows),
        "trigger_length": args.trigger_length,
        "copy_budget": args.copy_budget,
        "cooldown_tokens": args.cooldown_tokens,
        "reference_block_acceleration": args.reference_block_acceleration,
        "max_new_tokens": args.max_new_tokens,
        **speed_summary(token_rows, total_seconds, eos_token_id),
        **totals,
    }
    stats_path = output_path.with_name(output_path.stem + ".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved predictions -> {output_path}")
    print(f"saved stats -> {stats_path}")
    print(f"tokens_per_second_without_eos={summary['tokens_per_second_without_eos']:.2f}")


if __name__ == "__main__":
    main()
