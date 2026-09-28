#!/usr/bin/env python3
"""Offline-batched Base / CopyGen decoding with vLLM, with throughput per arm.

Each arm submits all prompts in one llm.generate call. tok/s is the number of
generated tokens (terminal EOS excluded) divided by the wall time of that call.
With --plugin the engine runs the CopyGen speculative worker (the Base arm then
runs in the same engine with the worker idle); without it, stock vLLM is used.

Example:
  python -m scripts.rebuttal.engine.vllm_batch \
    --model-path MODEL_PATH --data data/contract/test.jsonl --domain contract \
    --plugin PLUGIN_PT --arms base copygen --limit 24 \
    --output-dir outputs/rebuttal/engine/vllm_batch8/contract
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.rebuttal.engine.common import load_jsonl, render_editor_prompt, write_json, write_jsonl

WORKER_MODULE = "scripts.rebuttal.engine.vllm_copygen"


def build_llm(args):
    from vllm import LLM

    kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        enforce_eager=False,
        disable_log_stats=False,
        async_scheduling=False,
    )
    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.attention_backend != "auto":
        kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.plugin:
        os.environ["COPYGEN_VLLM_CONFIG"] = json.dumps(
            {
                "plugins": {args.domain: str(Path(args.plugin).resolve())},
                "min_match_length": 5,
                "max_candidate_starts": 8,
                "verifier_margin_floor": 0.0,
                "head_warmup_iterations": 3,
            }
        )
        kwargs.update(
            worker_cls=f"{WORKER_MODULE}.CopyGenWorker",
            spec_method="custom_class",
            spec_model=f"{WORKER_MODULE}.NoopProposer",
            spec_tokens=args.spec_tokens,
        )
    return LLM(**kwargs)


def sampling_params(SamplingParams, samples, arm: str, max_tokens: int, domain: str, stock: bool):
    if stock:
        return SamplingParams(
            temperature=0.0, top_k=1, max_tokens=max_tokens, ignore_eos=False, seed=0
        )
    return [
        SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=max_tokens,
            ignore_eos=False,
            seed=0,
            extra_args={
                "copygen_mode": arm,
                "copygen_domain": domain,
                "copygen_source_ids": sample["source_ids"],
            },
        )
        for sample in samples
    ]


def timed_generate(llm, prompts, params):
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    return outputs, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--plugin", default="", help="CopyGen plugin.pt; omit to run stock vLLM (Base only)"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=("base", "copygen"), default=["base", "copygen"]
    )
    parser.add_argument("--limit", type=int, default=0, help="Use the first N rows (0 = all)")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=0, help="0 keeps the vLLM default"
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--spec-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.plugin and "copygen" in args.arms:
        parser.error("the copygen arm needs --plugin")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = []
    for row in load_jsonl(args.data, limit=args.limit):
        prompt = render_editor_prompt(
            tokenizer, row["context"], row["instruction"], disable_thinking=args.disable_thinking
        )
        source_ids = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
        samples.append({"task_id": row["task_id"], "prompt": prompt, "source_ids": source_ids})

    from vllm import SamplingParams

    llm = build_llm(args)
    stock = not args.plugin
    prompts = [sample["prompt"] for sample in samples]

    if args.warmup_tokens:
        warmup = samples[: args.max_num_seqs]
        for arm in args.arms:
            params = sampling_params(
                SamplingParams, warmup, arm, args.warmup_tokens, args.domain, stock
            )
            timed_generate(llm, prompts[: len(warmup)], params)

    output_dir = Path(args.output_dir)
    stats = {"config": vars(args), "arms": {}}
    for arm in args.arms:
        params = sampling_params(SamplingParams, samples, arm, args.max_tokens, args.domain, stock)
        outputs, wall_seconds = timed_generate(llm, prompts, params)
        rows = []
        for sample, output in zip(samples, outputs):
            token_ids = list(output.outputs[0].token_ids)
            if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
            rows.append(
                {
                    "task_id": sample["task_id"],
                    "prediction": tokenizer.decode(token_ids, skip_special_tokens=True),
                    "generated_token_ids": token_ids,
                    "output_tokens": len(token_ids),
                }
            )
        write_jsonl(output_dir / f"{arm}.jsonl", rows)
        output_tokens = sum(row["output_tokens"] for row in rows)
        stats["arms"][arm] = {
            "requests": len(rows),
            "output_tokens": output_tokens,
            "wall_seconds": wall_seconds,
            "tokens_per_second": output_tokens / wall_seconds,
        }
        print(
            f"{arm}: {len(rows)} requests, {output_tokens} tokens, {wall_seconds:.2f}s, "
            f"{output_tokens / wall_seconds:.2f} tok/s",
            flush=True,
        )
    write_json(output_dir / "stats.json", stats)


if __name__ == "__main__":
    main()
