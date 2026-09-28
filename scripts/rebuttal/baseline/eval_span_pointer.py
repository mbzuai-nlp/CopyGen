#!/usr/bin/env python3
"""
Greedy decoding with a trained span-pointer checkpoint. At every step the best Gen(v) and the
best Copy(i:j) action are compared under the gate; a copied span is emitted as a whole.

Example:
  python scripts/rebuttal/baseline/eval_span_pointer.py \
    --checkpoint outputs/rebuttal/span_pointer/qwen3_8b/contract/span_pointer.pt \
    --oracle data/contract/test.jsonl \
    --output outputs/rebuttal/span_pointer/qwen3_8b/contract/predictions.jsonl \
    --max-actions 4096 \
    --max-output-tokens 4096 \
    --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv
from model.prompt_utils import encode_editor_prefix

apply_cuda_visible_devices_from_argv(sys.argv)

import torch
from tqdm.auto import tqdm

from scripts.rebuttal.baseline.span_pointer import SpanPointerModel, all_span_logsumexp, best_span


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


@torch.inference_mode()
def decode_one(
    model: SpanPointerModel,
    row: dict,
    *,
    device: torch.device,
    max_actions: int,
    max_output_tokens: int,
    disable_thinking: bool,
) -> tuple[str, list[int], dict]:
    tokenizer = model.tokenizer
    eos_id = tokenizer.eos_token_id

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    source_ids = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
    source_states = model.encode_source(torch.tensor([source_ids], dtype=torch.long, device=device))
    prefix_ids = encode_editor_prefix(
        tokenizer, row["context"], row["instruction"], device, disable_thinking=disable_thinking
    )
    hidden, past_key_values = model.hidden_states(prefix_ids, use_cache=True)
    hidden_state = hidden[:, -1, :]

    output_ids: list[int] = []
    copy_lengths: list[int] = []
    gen_actions = 0
    p_copy_sum = 0.0
    scored_actions = 0
    stop_reason = "max_actions"
    for _ in range(max_actions):
        if len(output_ids) >= max_output_tokens:
            stop_reason = "max_output_tokens"
            break

        gate_log_probs = torch.log_softmax(model.gate_logits(hidden_state)[0], dim=-1)
        best_gen_logprob, best_gen = torch.max(torch.log_softmax(model.vocab_logits(hidden_state)[0], dim=-1), dim=0)
        best_gen_score = gate_log_probs[0] + best_gen_logprob
        best_gen_id = int(best_gen.item())
        p_copy_sum += float(gate_log_probs[1].exp().item())
        scored_actions += 1

        # a span log-probability is at most 0, so COPY can only win if log p(copy) beats GEN
        copy_wins = False
        if bool((gate_log_probs[1] > best_gen_score).item()):
            start_scores, end_scores = model.endpoint_scores(hidden_state, source_states)
            span_score, copy_start, copy_end = best_span(start_scores[0], end_scores[0])
            best_copy_score = gate_log_probs[1] + span_score - all_span_logsumexp(start_scores[0], end_scores[0])
            copy_wins = bool(best_copy_score > best_gen_score)

        if copy_wins:
            copied_ids = source_ids[copy_start : copy_end + 1]
            if len(copied_ids) > max_output_tokens - len(output_ids):
                stop_reason = "max_output_tokens_before_atomic_copy"
                break
            # feed the copied span through the frozen LM once to extend the cache
            hidden, past_key_values = model.hidden_states(
                torch.tensor([copied_ids], dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_state = hidden[:, -1, :]
            output_ids.extend(copied_ids)
            copy_lengths.append(len(copied_ids))
            continue

        output_ids.append(best_gen_id)
        gen_actions += 1
        if best_gen_id == eos_id:
            stop_reason = "eos"
            break
        hidden, past_key_values = model.hidden_states(
            torch.tensor([[best_gen_id]], dtype=torch.long, device=device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_state = hidden[:, -1, :]

    prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
    torch.cuda.synchronize()
    stats = {
        "generation_seconds": time.perf_counter() - start_time,
        "generated_tokens": len(output_ids),
        "action_count": gen_actions + len(copy_lengths),
        "scored_action_count": scored_actions,
        "gen_action_count": gen_actions,
        "copy_action_count": len(copy_lengths),
        "copied_token_count": sum(copy_lengths),
        "copy_lengths": copy_lengths,
        "mean_p_copy": p_copy_sum / scored_actions if scored_actions else 0.0,
        "stop_reason": stop_reason,
    }
    return prediction, output_ids, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy decoding with a frozen-LM span-pointer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", default="", help="Defaults to the base model recorded in the checkpoint.")
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_path = args.model_path or checkpoint["model_path"]
    device = torch.device("cuda")
    model = SpanPointerModel(model_path, score_scale=checkpoint["score_scale"])
    model.load_heads(checkpoint)
    model.to(device)
    model.eval()
    print(
        f"model={model_path} checkpoint={args.checkpoint} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    rows = load_rows(Path(args.oracle), limit=args.max_samples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    stop_reasons = Counter()
    eos_rows = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="span-pointer", unit="sample"):
            prediction, output_ids, stats = decode_one(
                model,
                row,
                device=device,
                max_actions=args.max_actions,
                max_output_tokens=args.max_output_tokens,
                disable_thinking=args.disable_thinking,
            )
            for key in ("generated_tokens", "generation_seconds", "action_count", "copy_action_count", "copied_token_count"):
                totals[key] += stats[key]
            stop_reasons[stats["stop_reason"]] += 1
            eos_rows += stats["stop_reason"] == "eos"
            out_row = dict(row)
            out_row["prediction"] = prediction
            out_row["model_edited_answer"] = prediction
            out_row["model_name"] = model_path
            out_row["generated_token_ids"] = output_ids
            out_row["span_pointer_stats"] = stats
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    seconds = totals["generation_seconds"]
    summary = {
        "model_path": model_path,
        "checkpoint": args.checkpoint,
        "oracle": args.oracle,
        "sample_count": len(rows),
        "max_actions": args.max_actions,
        "max_output_tokens": args.max_output_tokens,
        **totals,
        "tokens_per_second": totals["generated_tokens"] / seconds if seconds else 0.0,
        "tokens_per_second_without_eos": (totals["generated_tokens"] - eos_rows) / seconds if seconds else 0.0,
        "copy_token_fraction": (
            totals["copied_token_count"] / totals["generated_tokens"] if totals["generated_tokens"] else 0.0
        ),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
    }
    stats_path = output_path.with_name(output_path.stem + ".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved predictions -> {output_path}")
    print(f"saved stats -> {stats_path}")
    print(f"tokens_per_second={summary['tokens_per_second']:.2f}")


if __name__ == "__main__":
    main()
