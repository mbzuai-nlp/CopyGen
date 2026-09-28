#!/usr/bin/env python3
"""
Greedy (or sampled) baseline with the same prompt but without copy heads.

Example:
  python scripts/baseline/eval_llm_baseline.py \
  --model-path MODEL_PATH \
  --oracle data/random_sequence/test.jsonl \
  --output outputs/predictions/random_sequence_llm_baseline.jsonl \
  --cuda-visible-devices 0 \
  --device auto \
  --disable-thinking

  python scripts/baseline/eval_llm_baseline.py \
  --model-path MODEL_PATH \
  --oracle data/random_sequence/test.jsonl \
  --output outputs/predictions/random_sequence_llm_baseline_alt.jsonl \
  --cuda-visible-devices 1 \
  --device auto

  # token-by-token decoding with the CopyGen evaluator's sampler and EOS handling
  python scripts/baseline/eval_llm_baseline.py \
  --model-path MODEL_PATH \
  --oracle data/contract/test.jsonl \
  --output outputs/predictions/contract_llm_baseline_top_p.jsonl \
  --cuda-visible-devices 0 \
  --device cuda \
  --stepwise --temperature 0.7 --top-p 0.9 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _apply_cuda_visible_devices_from_argv(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda-visible-devices", default=None)
    args, _ = parser.parse_known_args(argv[1:])
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


_apply_cuda_visible_devices_from_argv(sys.argv)

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.prompt_utils import encode_editor_prefix
from model.sampling_utils import select_next_token, validate_sampling_args


DEFAULT_MODEL_PATH = "MODEL_PATH"


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


def _get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


def _load_model(model_path: str, tokenizer, device_mode: str):
    use_cuda = torch.cuda.is_available() and device_mode != "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    def _from_pretrained(**kwargs):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=dtype,
                **kwargs,
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                **kwargs,
            )

    if device_mode == "auto" and torch.cuda.is_available():
        try:
            model = _from_pretrained(device_map="auto", low_cpu_mem_usage=True)
            return model, _get_model_input_device(model), "auto", dtype
        except Exception:
            model = _from_pretrained(low_cpu_mem_usage=True)
            model.to("cpu")
            return model, torch.device("cpu"), "cpu-fallback", torch.float32

    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")
        model = _from_pretrained(low_cpu_mem_usage=True)
        model.to("cuda")
        return model, torch.device("cuda"), "cuda", dtype

    model = _from_pretrained(low_cpu_mem_usage=True)
    model.to("cpu")
    return model, torch.device("cpu"), "cpu", torch.float32


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    row: dict,
    device: torch.device,
    max_new_tokens: int,
    skip_special_tokens: bool,
    disable_thinking: bool,
) -> tuple[str, list[int]]:
    prefix_ids = encode_editor_prefix(
        tokenizer,
        row["context"],
        row["instruction"],
        device,
        disable_thinking=disable_thinking,
    )
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
    text = tokenizer.decode(out_ids, skip_special_tokens=skip_special_tokens)
    return text, out_ids


@torch.no_grad()
def generate_one_stepwise(
    model,
    tokenizer,
    row: dict,
    device: torch.device,
    max_new_tokens: int,
    skip_special_tokens: bool,
    disable_thinking: bool,
    temperature: float,
    top_p: float,
    min_p: float,
    generator: torch.Generator,
) -> tuple[str, list[int]]:
    """Decode one token at a time like model/eval_copy_constrained.py: same sampler, stop before EOS."""
    decoder = model.get_decoder()
    prefix_ids = encode_editor_prefix(
        tokenizer,
        row["context"],
        row["instruction"],
        device,
        disable_thinking=disable_thinking,
    )
    output = decoder(input_ids=prefix_ids, use_cache=True)
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        hidden_state = output.last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden_state.to(model.lm_head.weight.dtype))[0]
        token_id = select_next_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            generator=generator,
        )
        if token_id == tokenizer.eos_token_id:
            break
        out_ids.append(token_id)
        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        output = decoder(input_ids=token_tensor, past_key_values=output.past_key_values, use_cache=True)
    text = tokenizer.decode(out_ids, skip_special_tokens=skip_special_tokens)
    return text, out_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy same-prompt baseline without copy heads.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--oracle", default="data/random_sequence/test.jsonl")
    parser.add_argument(
        "--output",
        default="outputs/predictions/random_sequence_llm_baseline.jsonl",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--skip-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stepwise",
        action="store_true",
        help="Decode token by token with model.sampling_utils instead of model.generate (required for sampling).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="Sample i is decoded with seed + i.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '1'. Applied before torch import.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    try:
        validate_sampling_args(args.temperature, args.top_p, args.min_p)
    except ValueError as exc:
        parser.error(str(exc))
    if args.temperature > 0 and not args.stepwise:
        parser.error("sampling (--temperature > 0) requires --stepwise")

    rows = load_rows(Path(args.oracle), limit=args.max_samples)
    if not rows:
        raise SystemExit("No rows found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, device, load_mode, dtype = _load_model(args.model_path, tokenizer, args.device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_generated_tokens = 0
    total_generation_seconds = 0.0

    print(
        f"loaded baseline model from {args.model_path} "
        f"(load_mode={load_mode}, dtype={dtype}, disable_thinking={args.disable_thinking}, "
        f"stepwise={args.stepwise}, temperature={args.temperature}, top_p={args.top_p}, min_p={args.min_p}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
    )

    with output_path.open("w", encoding="utf-8") as f:
        for sample_idx, row in enumerate(tqdm(rows, desc="baseline", unit="sample")):
            start_time = time.perf_counter()
            if args.stepwise:
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + sample_idx)
                pred_text, pred_ids = generate_one_stepwise(
                    model=model,
                    tokenizer=tokenizer,
                    row=row,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    skip_special_tokens=args.skip_special_tokens,
                    disable_thinking=args.disable_thinking,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    generator=generator,
                )
            else:
                pred_text, pred_ids = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    row=row,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    skip_special_tokens=args.skip_special_tokens,
                    disable_thinking=args.disable_thinking,
                )
            total_generation_seconds += time.perf_counter() - start_time
            total_generated_tokens += len(pred_ids)
            out_row = dict(row)
            out_row["prediction"] = pred_text
            out_row["model_edited_answer"] = pred_text
            out_row["model_name"] = args.model_path
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
        "stepwise": args.stepwise,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
    }
    speed_path = output_path.with_name(output_path.stem + ".speed.json")
    speed_path.write_text(json.dumps(speed_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved predictions -> {output_path}")
    print(f"saved speed stats -> {speed_path}")
    print(f"avg_tokens_per_second={avg_tokens_per_second:.4f}")


if __name__ == "__main__":
    main()
