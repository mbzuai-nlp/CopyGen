#!/usr/bin/env python3
"""
Rule-based Longest-Continuation baseline: decode greedily until the last `trigger_length`
output tokens match the input context, pick the matching occurrence with the longest backward
match (ties: longer remaining continuation) and copy the rest of the context without
verification. Decoding stops after the copy.

Example:
  python scripts/rebuttal/baseline/eval_longest_copy.py \
    --model-path MODEL_PATH \
    --oracle data/contract/test.jsonl \
    --output outputs/rebuttal/baselines/qwen3_8b/contract/longest_copy.jsonl \
    --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv
from model.prompt_utils import encode_editor_prefix

apply_cuda_visible_devices_from_argv(sys.argv)

import torch
from tqdm.auto import tqdm

from scripts.rebuttal.baseline.eval_copyspec import forward, load_model, load_rows
from scripts.rebuttal.baseline.eval_picard import build_source_index, grounded_states


def backward_match_length(generated: list[int], reference: list[int], position: int) -> int:
    """Number of output tokens that match the context right before `position`."""
    limit = min(len(generated), position)
    length = 0
    while length < limit and generated[-length - 1] == reference[position - length - 1]:
        length += 1
    return length


@torch.no_grad()
def longest_copy_decode(
    model,
    prefix_ids: torch.Tensor,
    reference: list[int],
    index: dict,
    *,
    eos_token_id: int,
    max_new_tokens: int,
    trigger_length: int,
) -> tuple[list[int], dict | None]:
    outputs = forward(model, prefix_ids)
    past_key_values = outputs.past_key_values
    generated = [int(outputs.logits[0, -1].argmax())]

    while len(generated) < max_new_tokens and generated[-1] != eos_token_id:
        states = grounded_states(index, generated, reference, trigger_length)
        if states:
            matches = {state: backward_match_length(generated, reference, state) for state in states}
            start = max(states, key=lambda state: (matches[state], -state))
            continuation = reference[start:][: max_new_tokens - len(generated)]
            if eos_token_id in continuation:
                continuation = continuation[: continuation.index(eos_token_id) + 1]
            generated.extend(continuation)
            copy = {
                "candidates": len(states),
                "copy_start": start,
                "backward_match_length": matches[start],
                "copied_tokens": len(continuation),
            }
            return generated[:max_new_tokens], copy

        block = torch.tensor([[generated[-1]]], dtype=torch.long, device=prefix_ids.device)
        outputs = forward(model, block, past_key_values)
        past_key_values = outputs.past_key_values
        generated.append(int(outputs.logits[0, -1].argmax()))

    return generated[:max_new_tokens], None


def main() -> None:
    parser = argparse.ArgumentParser(description="Longest-Continuation copy baseline without verification.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--trigger-length", type=int, default=5)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    rows = load_rows(Path(args.oracle), limit=args.max_samples)
    tokenizer, model = load_model(args.model_path)
    device = torch.device("cuda")
    print(
        f"loaded {args.model_path} trigger={args.trigger_length} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    triggered = 0
    copied_tokens = 0
    generated_tokens = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="longest-copy", unit="sample"):
            prefix_ids = encode_editor_prefix(
                tokenizer, row["context"], row["instruction"], device, disable_thinking=args.disable_thinking
            )
            reference = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
            token_ids, copy = longest_copy_decode(
                model,
                prefix_ids,
                reference,
                build_source_index(reference, args.trigger_length),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                trigger_length=args.trigger_length,
            )
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True)

            generated_tokens += len(token_ids)
            if copy is not None:
                triggered += 1
                copied_tokens += copy["copied_tokens"]
            out_row = dict(row)
            out_row["prediction"] = prediction
            out_row["model_edited_answer"] = prediction
            out_row["model_name"] = args.model_path
            out_row["generated_token_ids"] = token_ids
            out_row["longest_copy"] = copy
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "model_path": args.model_path,
        "oracle": args.oracle,
        "sample_count": len(rows),
        "trigger_length": args.trigger_length,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "triggered_rows": triggered,
        "copied_tokens": copied_tokens,
    }
    stats_path = output_path.with_name(output_path.stem + ".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved predictions -> {output_path}")
    print(f"saved stats -> {stats_path}")


if __name__ == "__main__":
    main()
