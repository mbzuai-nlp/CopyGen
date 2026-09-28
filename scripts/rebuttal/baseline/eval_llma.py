#!/usr/bin/env python3
"""
LLMA / Inference-with-Reference baseline (Yang et al., 2023): match the output suffix against
the input context, draft the following context tokens and verify them greedily.

With the authors' n=1, K=20 setting, suffixes of length 1..5 are matched and a verification
block holds the last output token plus at most K-1 drafted tokens.

Example:
  python scripts/rebuttal/baseline/eval_llma.py \
    --model-path MODEL_PATH \
    --oracle data/contract/test.jsonl \
    --output outputs/rebuttal/baselines/qwen3_8b/contract/llma.jsonl \
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

from scripts.rebuttal.baseline.eval_copyspec import accept_draft, forward, load_model, load_rows, speed_summary


def build_reference_index(reference: list[int], min_length: int, max_length: int) -> dict:
    index = {length: {} for length in range(min_length, max_length + 1)}
    for length, table in index.items():
        for start in range(len(reference) - length + 1):
            table.setdefault(tuple(reference[start : start + length]), []).append(start)
    return index


def longest_reference_match(
    index: dict, generated: list[int], min_length: int, max_length: int
) -> tuple[int, int] | None:
    """Longest output suffix found in the reference; ties go to the first occurrence."""
    for length in range(min(max_length, len(generated)), min_length - 1, -1):
        starts = index[length].get(tuple(generated[-length:]))
        if starts:
            return starts[0], length
    return None


@torch.no_grad()
def llma_decode(
    model,
    prefix_ids: torch.Tensor,
    reference: list[int],
    index: dict,
    *,
    eos_token_id: int,
    max_new_tokens: int,
    min_match_length: int,
    max_match_length: int,
    block_size: int,
) -> tuple[list[int], dict]:
    device = prefix_ids.device
    prompt_length = prefix_ids.shape[1]
    stats = {
        "reference_lookups": 0,
        "reference_matches": 0,
        "copy_attempts": 0,
        "copy_proposed_tokens": 0,
        "copy_accepted_tokens": 0,
        "matched_suffix_tokens": 0,
        "longest_matched_suffix": 0,
        "target_forward_calls": 1,
    }

    outputs = forward(model, prefix_ids)
    past_key_values = outputs.past_key_values
    generated = [int(outputs.logits[0, -1].argmax())]

    while len(generated) < max_new_tokens and generated[-1] != eos_token_id:
        remaining = max_new_tokens - len(generated)
        stats["reference_lookups"] += 1
        match = longest_reference_match(index, generated, min_match_length, max_match_length)
        draft = []
        if match is not None:
            start, length = match
            stats["reference_matches"] += 1
            stats["matched_suffix_tokens"] += length
            stats["longest_matched_suffix"] = max(stats["longest_matched_suffix"], length)
            draft = reference[start + length : start + length + min(block_size - 1, remaining - 1)]

        stats["target_forward_calls"] += 1
        if draft:
            length_before = prompt_length + len(generated)
            block = torch.tensor([[generated[-1], *draft]], dtype=torch.long, device=device)
            outputs = forward(model, block, past_key_values, all_logits=True)
            past_key_values = outputs.past_key_values
            emitted, accepted = accept_draft(draft, outputs.logits[0].argmax(dim=-1).tolist(), eos_token_id)
            past_key_values.crop(length_before + len(emitted) - 1)
            stats["copy_attempts"] += 1
            stats["copy_proposed_tokens"] += len(draft)
            stats["copy_accepted_tokens"] += accepted
        else:
            block = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            outputs = forward(model, block, past_key_values)
            past_key_values = outputs.past_key_values
            emitted = [int(outputs.logits[0, -1].argmax())]
        generated.extend(emitted)

    return generated[:max_new_tokens], stats


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMA reference-draft baseline with greedy verification.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--min-match-length", type=int, default=1)
    parser.add_argument("--max-match-length", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    rows = load_rows(Path(args.oracle), limit=args.max_samples)
    tokenizer, model = load_model(args.model_path)
    device = torch.device("cuda")
    eos_token_id = tokenizer.eos_token_id
    print(
        f"loaded {args.model_path} match={args.min_match_length}-{args.max_match_length} "
        f"block_size={args.block_size} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_rows: list[list[int]] = []
    totals: dict[str, int] = {}
    total_seconds = 0.0
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="llma", unit="sample"):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            prefix_ids = encode_editor_prefix(
                tokenizer, row["context"], row["instruction"], device, disable_thinking=args.disable_thinking
            )
            reference = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
            index = build_reference_index(reference, args.min_match_length, args.max_match_length)
            token_ids, stats = llma_decode(
                model,
                prefix_ids,
                reference,
                index,
                eos_token_id=eos_token_id,
                max_new_tokens=args.max_new_tokens,
                min_match_length=args.min_match_length,
                max_match_length=args.max_match_length,
                block_size=args.block_size,
            )
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start_time

            total_seconds += seconds
            token_rows.append(token_ids)
            for key, value in stats.items():
                if key == "longest_matched_suffix":
                    totals[key] = max(totals.get(key, 0), value)
                else:
                    totals[key] = totals.get(key, 0) + value
            out_row = dict(row)
            out_row["prediction"] = prediction
            out_row["model_edited_answer"] = prediction
            out_row["model_name"] = args.model_path
            out_row["generated_token_ids"] = token_ids
            out_row["llma_stats"] = stats
            out_row["generation_seconds"] = seconds
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "model_path": args.model_path,
        "oracle": args.oracle,
        "sample_count": len(rows),
        "min_match_length": args.min_match_length,
        "max_match_length": args.max_match_length,
        "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
        **speed_summary(token_rows, total_seconds, eos_token_id),
        **totals,
        "copy_acceptance_ratio": (
            totals["copy_accepted_tokens"] / totals["copy_proposed_tokens"] if totals.get("copy_proposed_tokens") else 0.0
        ),
    }
    stats_path = output_path.with_name(output_path.stem + ".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved predictions -> {output_path}")
    print(f"saved stats -> {stats_path}")
    print(f"tokens_per_second_without_eos={summary['tokens_per_second_without_eos']:.2f}")


if __name__ == "__main__":
    main()
