#!/usr/bin/env python3
"""
CopySpec baseline (Dumitru et al., 2025): after the last gamma tokens match an earlier
position of prompt + output, draft the tokens that followed it and verify them greedily.

Example:
  python scripts/rebuttal/baseline/eval_copyspec.py \
    --model-path MODEL_PATH \
    --oracle data/contract/test.jsonl \
    --output outputs/rebuttal/baselines/qwen3_8b/contract/copyspec.jsonl \
    --gamma 3 \
    --copy-length 10 \
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
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    return tokenizer, model


def forward(model, input_ids: torch.Tensor, past_key_values=None, all_logits: bool = False):
    """Cached forward over new tokens, with explicit positions after the cached prefix."""
    past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
    query_length = input_ids.shape[1]
    positions = torch.arange(past_length, past_length + query_length, device=input_ids.device)
    return model(
        input_ids=input_ids,
        attention_mask=torch.ones((1, past_length + query_length), dtype=torch.long, device=input_ids.device),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=query_length if all_logits else 1,
        return_dict=True,
    )


def accept_draft(draft: list[int], greedy: list[int], eos_token_id: int) -> tuple[list[int], int]:
    """Keep the draft prefix that equals the greedy tokens, then add the next greedy token.

    `greedy[i]` is the greedy token after the i-th input of the verification block, so it has
    one more entry than the draft.
    """
    accepted = 0
    for token, greedy_token in zip(draft, greedy):
        if token != greedy_token:
            break
        accepted += 1
        if token == eos_token_id:
            return draft[:accepted], accepted
    return draft[:accepted] + [greedy[accepted]], accepted


def speed_summary(token_rows: list[list[int]], seconds: float, eos_token_id: int) -> dict:
    generated = sum(len(ids) for ids in token_rows)
    eos_rows = sum(bool(ids) and ids[-1] == eos_token_id for ids in token_rows)
    return {
        "generated_tokens": generated,
        "eos_terminated_rows": eos_rows,
        "generation_seconds": seconds,
        "tokens_per_second": generated / seconds if seconds else 0.0,
        # reported throughput: one terminal EOS per EOS-terminated row is not counted
        "tokens_per_second_without_eos": (generated - eos_rows) / seconds if seconds else 0.0,
    }


def add_ngrams(index: dict, tokens: list[int], gamma: int, next_start: int) -> int:
    end = max(next_start, len(tokens) - gamma + 1)
    for start in range(next_start, end):
        index.setdefault(tuple(tokens[start : start + gamma]), []).append(start)
    return end


def earliest_match(index: dict, tokens: list[int], gamma: int) -> int | None:
    for start in index.get(tuple(tokens[-gamma:]), ()):
        # overlap with the current suffix is allowed, the suffix itself is not
        if start + gamma < len(tokens):
            return start
    return None


@torch.no_grad()
def copyspec_decode(
    model,
    prefix_ids: torch.Tensor,
    *,
    eos_token_id: int,
    max_new_tokens: int,
    gamma: int,
    copy_length: int,
) -> tuple[list[int], dict]:
    device = prefix_ids.device
    tokens = prefix_ids[0].tolist()
    prompt_length = len(tokens)
    index: dict[tuple[int, ...], list[int]] = {}
    indexed = add_ngrams(index, tokens, gamma, 0)
    stats = {"copy_attempts": 0, "copy_proposed_tokens": 0, "copy_accepted_tokens": 0, "target_forward_calls": 1}

    outputs = forward(model, prefix_ids)
    past_key_values = outputs.past_key_values
    tokens.append(int(outputs.logits[0, -1].argmax()))
    indexed = add_ngrams(index, tokens, gamma, indexed)

    while len(tokens) - prompt_length < max_new_tokens and tokens[-1] != eos_token_id:
        remaining = max_new_tokens - (len(tokens) - prompt_length)
        start = earliest_match(index, tokens, gamma)
        draft = []
        if start is not None and copy_length > 0 and remaining > 1:
            draft = tokens[start + gamma : start + gamma + min(copy_length, remaining - 1)]

        stats["target_forward_calls"] += 1
        if draft:
            # the cache holds every token except the last one, which starts the block
            length_before = len(tokens)
            block = torch.tensor([[tokens[-1], *draft]], dtype=torch.long, device=device)
            outputs = forward(model, block, past_key_values, all_logits=True)
            past_key_values = outputs.past_key_values
            emitted, accepted = accept_draft(draft, outputs.logits[0].argmax(dim=-1).tolist(), eos_token_id)
            past_key_values.crop(length_before + len(emitted) - 1)
            stats["copy_attempts"] += 1
            stats["copy_proposed_tokens"] += len(draft)
            stats["copy_accepted_tokens"] += accepted
        else:
            block = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            outputs = forward(model, block, past_key_values)
            past_key_values = outputs.past_key_values
            emitted = [int(outputs.logits[0, -1].argmax())]
        tokens.extend(emitted)
        indexed = add_ngrams(index, tokens, gamma, indexed)

    return tokens[prompt_length : prompt_length + max_new_tokens], stats


def main() -> None:
    parser = argparse.ArgumentParser(description="CopySpec baseline with greedy verification.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--copy-length", type=int, default=10)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    rows = load_rows(Path(args.oracle), limit=args.max_samples)
    tokenizer, model = load_model(args.model_path)
    device = torch.device("cuda")
    eos_token_id = tokenizer.eos_token_id
    print(
        f"loaded {args.model_path} gamma={args.gamma} copy_length={args.copy_length} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_rows: list[list[int]] = []
    totals = {"copy_attempts": 0, "copy_proposed_tokens": 0, "copy_accepted_tokens": 0, "target_forward_calls": 0}
    total_seconds = 0.0
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="copyspec", unit="sample"):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            prefix_ids = encode_editor_prefix(
                tokenizer, row["context"], row["instruction"], device, disable_thinking=args.disable_thinking
            )
            token_ids, stats = copyspec_decode(
                model,
                prefix_ids,
                eos_token_id=eos_token_id,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                copy_length=args.copy_length,
            )
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start_time

            total_seconds += seconds
            token_rows.append(token_ids)
            for key in totals:
                totals[key] += stats[key]
            out_row = dict(row)
            out_row["prediction"] = prediction
            out_row["model_edited_answer"] = prediction
            out_row["model_name"] = args.model_path
            out_row["generated_token_ids"] = token_ids
            out_row["copyspec_stats"] = stats
            out_row["generation_seconds"] = seconds
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "model_path": args.model_path,
        "oracle": args.oracle,
        "sample_count": len(rows),
        "gamma": args.gamma,
        "copy_length": args.copy_length,
        "max_new_tokens": args.max_new_tokens,
        **speed_summary(token_rows, total_seconds, eos_token_id),
        **totals,
        "copy_acceptance_ratio": (
            totals["copy_accepted_tokens"] / totals["copy_proposed_tokens"] if totals["copy_proposed_tokens"] else 0.0
        ),
    }
    stats_path = output_path.with_name(output_path.stem + ".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved predictions -> {output_path}")
    print(f"saved stats -> {stats_path}")
    print(f"tokens_per_second_without_eos={summary['tokens_per_second_without_eos']:.2f}")


if __name__ == "__main__":
    main()
