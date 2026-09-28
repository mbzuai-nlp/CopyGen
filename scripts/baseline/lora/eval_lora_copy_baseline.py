#!/usr/bin/env python3
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

apply_cuda_visible_devices_from_argv(sys.argv, default="1")

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from model.prompt_utils import encode_editor_prefix
except ModuleNotFoundError:
    from prompt_utils import encode_editor_prefix


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


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    row: dict,
    device: torch.device,
    max_new_tokens: int,
    skip_special_tokens: bool,
) -> tuple[str, list[int]]:
    prefix_ids = encode_editor_prefix(tokenizer, row["context"], row["instruction"], device)
    attention_mask = torch.ones_like(prefix_ids, device=device)
    generated = model.generate(
        input_ids=prefix_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    out_ids = generated[0, prefix_ids.shape[1] :].tolist()
    return tokenizer.decode(out_ids, skip_special_tokens=skip_special_tokens), out_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA baseline on copy tasks.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--skip-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '1'. Applied before torch import.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.oracle), limit=args.max_samples)
    if not rows:
        raise SystemExit("No rows found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            dtype=dtype,
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.to(device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_generated_tokens = 0
    total_generation_seconds = 0.0
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="lora-baseline", unit="sample"):
            start_time = time.perf_counter()
            pred_text, pred_ids = generate_one(
                model=model,
                tokenizer=tokenizer,
                row=row,
                device=device,
                max_new_tokens=args.max_new_tokens,
                skip_special_tokens=args.skip_special_tokens,
            )
            total_generation_seconds += time.perf_counter() - start_time
            total_generated_tokens += len(pred_ids)
            out_row = dict(row)
            out_row["prediction"] = pred_text
            out_row["model_edited_answer"] = pred_text
            out_row["model_name"] = f"{args.model_path}+LoRA"
            out_row["generated_token_ids"] = pred_ids
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    avg_tokens_per_second = (
        total_generated_tokens / total_generation_seconds if total_generation_seconds > 0 else 0.0
    )
    speed_stats = {
        "sample_count": len(rows),
        "total_generated_tokens": total_generated_tokens,
        "total_generation_seconds": total_generation_seconds,
        "avg_tokens_per_second": avg_tokens_per_second,
    }
    speed_path = output_path.with_name(output_path.stem + ".speed.json")
    speed_path.write_text(json.dumps(speed_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved predictions -> {output_path}")
    print(f"saved speed stats -> {speed_path}")
    print(f"avg_tokens_per_second={avg_tokens_per_second:.4f}")


if __name__ == "__main__":
    main()
