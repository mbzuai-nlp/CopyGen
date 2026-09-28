#!/usr/bin/env python3
"""Paired Base / CopyGen decoding with SGLang, one request at a time.

Base is a single greedy Engine.generate request. CopyGen runs the control loop
in the client (sglang_copygen.decode_copygen) on SGLang's hidden states and
log-probs. The radix cache is flushed before every measured arm; wall_seconds
is the time of one arm for one row.

Example:
  python -m scripts.rebuttal.engine.sglang_client \
    --model-path MODEL_PATH --data data/contract/test.jsonl --plugin PLUGIN_PT \
    --output-dir outputs/rebuttal/engine/sglang_client/contract
"""

from __future__ import annotations

import os

os.environ.setdefault("SGLANG_IS_FLASHINFER_AVAILABLE", "false")

import argparse
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.rebuttal.engine.common import (
    SourceMatcher,
    load_heads,
    load_jsonl,
    render_editor_prompt,
    write_json,
    write_jsonl,
)
from scripts.rebuttal.engine.sglang_copygen import ScoreBackend, decode_copygen

MODES = ("base", "copygen")


def run_base(engine, tokenizer, prompt_ids, max_output_tokens):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = engine.generate(
        input_ids=prompt_ids,
        sampling_params={
            "temperature": 0.0,
            "top_k": 1,
            "max_new_tokens": max_output_tokens,
            "ignore_eos": False,
            "skip_special_tokens": False,
        },
        return_logprob=False,
        return_hidden_states=False,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    token_ids = [int(token_id) for token_id in output["output_ids"]]
    if token_ids and token_ids[-1] == tokenizer.eos_token_id:
        token_ids = token_ids[:-1]
    return {"generated_token_ids": token_ids, "wall_seconds": wall_seconds}


def run_copygen(engine, heads, tokenizer, sample, max_steps, max_output_tokens, chunk_size):
    backend = ScoreBackend(
        engine, hidden_size=heads.plugin_input_norm.weight.numel() - 1, next_top_k=9
    )
    matcher = SourceMatcher(sample["source_ids"], tokenizer=tokenizer)
    torch.cuda.synchronize()
    started = time.perf_counter()
    decoded = decode_copygen(
        backend,
        heads,
        tokenizer,
        sample["prompt_ids"],
        sample["source_ids"],
        matcher,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        chunk_size=chunk_size,
    )
    torch.cuda.synchronize()
    decoded["wall_seconds"] = time.perf_counter() - started
    decoded["engine_requests"] = backend.request_count
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument(
        "--chunk-size", type=int, default=17, help="Proposal tokens verified in the first prefill"
    )
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--mem-fraction-static", type=float, default=0.72)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    samples = []
    for row in load_jsonl(args.data):
        prompt = render_editor_prompt(
            tokenizer, row["context"], row["instruction"], disable_thinking=args.disable_thinking
        )
        samples.append(
            {
                "task_id": row["task_id"],
                "prompt_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
                "source_ids": tokenizer(row["context"], add_special_tokens=False)["input_ids"],
            }
        )

    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        random_seed=args.seed,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        max_running_requests=1,
        tp_size=1,
        schedule_policy="fcfs",
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        enable_return_hidden_states=True,
        enable_metrics=False,
        attention_backend="triton",
        sampling_backend="pytorch",
        log_level="error",
    )
    results = {mode: [] for mode in MODES}
    try:
        heads = load_heads(args.plugin, torch.device("cuda:0"))
        if args.warmup_tokens:
            engine.flush_cache()
            run_base(engine, tokenizer, samples[0]["prompt_ids"], args.warmup_tokens)
            engine.flush_cache()
            run_copygen(
                engine,
                heads,
                tokenizer,
                samples[0],
                args.warmup_tokens,
                args.warmup_tokens,
                args.chunk_size,
            )
            engine.flush_cache()

        # Alternate the arm order per row so neither arm always runs first.
        rng = random.Random(args.seed)
        for index, sample in enumerate(samples):
            for mode in rng.sample(MODES, len(MODES)):
                engine.flush_cache()
                if mode == "base":
                    result = run_base(
                        engine, tokenizer, sample["prompt_ids"], args.max_output_tokens
                    )
                else:
                    result = run_copygen(
                        engine,
                        heads,
                        tokenizer,
                        sample,
                        args.max_steps,
                        args.max_output_tokens,
                        args.chunk_size,
                    )
                prediction = tokenizer.decode(
                    result["generated_token_ids"], skip_special_tokens=True
                )
                results[mode].append(
                    {"task_id": sample["task_id"], "prediction": prediction, **result}
                )
                tokens = len(result["generated_token_ids"])
                print(
                    f"[{index + 1}/{len(samples)}] {sample['task_id']} {mode} "
                    f"tokens={tokens} wall={result['wall_seconds']:.3f}s",
                    flush=True,
                )
    finally:
        engine.shutdown()

    output_dir = Path(args.output_dir)
    stats = {"config": vars(args), "arms": {}}
    for mode in MODES:
        rows = results[mode]
        write_jsonl(output_dir / f"{mode}.jsonl", rows)
        output_tokens = sum(len(row["generated_token_ids"]) for row in rows)
        wall_seconds = sum(row["wall_seconds"] for row in rows)
        stats["arms"][mode] = {
            "requests": len(rows),
            "output_tokens": output_tokens,
            "wall_seconds": wall_seconds,
            "tokens_per_second": output_tokens / wall_seconds,
        }
    write_json(output_dir / "stats.json", stats)


if __name__ == "__main__":
    main()
