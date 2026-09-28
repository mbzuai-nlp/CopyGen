#!/usr/bin/env python3
"""Batched SGLang decoding with the native COPYGEN speculative algorithm, or stock SGLang.

All rows go into one Engine.generate call (at most --max-running-requests at a
time) after two untimed warmup requests. tok/s is the number of generated tokens
(terminal EOS excluded) divided by the wall time of that call. With --plugin the
engine uses CopyGenNativeWorker and the stats also include the CUDA time of the
trained heads as a share of that wall time.

Example:
  python -m scripts.rebuttal.engine.sglang_native \
    --model-path MODEL_PATH --data data/contract/test.jsonl --limit 24 \
    --plugin PLUGIN_PT --draft-tokens 64 \
    --output-dir outputs/rebuttal/engine/sglang_native/contract_copygen
"""

from __future__ import annotations

import os

os.environ.setdefault("SGLANG_IS_FLASHINFER_AVAILABLE", "false")

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.rebuttal.engine.common import load_jsonl, render_editor_prompt, write_json, write_jsonl
from scripts.rebuttal.engine.sglang_copygen import register_copygen_algorithm

# SGLang starts its scheduler in a fresh process that re-imports this module, so
# the algorithm must be registered at import time.
register_copygen_algorithm()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--plugin", default="", help="CopyGen plugin.pt; omit to run stock SGLang")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=24, help="Use the first N rows (0 = all)")
    parser.add_argument(
        "--draft-tokens", type=int, default=64, help="Draft window including the root token"
    )
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--mem-fraction-static", type=float, default=0.78)
    parser.add_argument("--warmup-rows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    rows = load_jsonl(args.data, limit=args.limit)
    prompt_ids = [
        tokenizer(
            render_editor_prompt(
                tokenizer,
                row["context"],
                row["instruction"],
                disable_thinking=args.disable_thinking,
            ),
            add_special_tokens=False,
        )["input_ids"]
        for row in rows
    ]
    rids = [f"row{index:02d}-{row['task_id']}" for index, row in enumerate(rows)]

    from sglang.srt.entrypoints.engine import Engine

    sampling_params = {
        "temperature": 0.0,
        "top_k": 1,
        "max_new_tokens": args.max_output_tokens,
        "ignore_eos": False,
        "skip_special_tokens": False,
    }
    head_timing = None
    with tempfile.TemporaryDirectory() as work_dir:
        # Files shared with the worker, which runs in SGLang's scheduler process.
        head_timing_path = Path(work_dir) / "head_timing.json"
        engine_kwargs = {}
        if args.plugin:
            sources_path = Path(work_dir) / "source_ids.json"
            sources = {
                rid: tokenizer(row["context"], add_special_tokens=False)["input_ids"]
                for rid, row in zip(rids, rows)
            }
            sources_path.write_text(json.dumps(sources), encoding="utf-8")
            os.environ["COPYGEN_SGLANG_SOURCES"] = str(sources_path)
            os.environ["COPYGEN_SGLANG_PLUGIN"] = str(Path(args.plugin).resolve())
            os.environ["COPYGEN_SGLANG_STATS"] = str(head_timing_path)
            engine_kwargs = dict(
                speculative_algorithm="COPYGEN",
                speculative_num_draft_tokens=args.draft_tokens,
                speculative_eagle_topk=1,
                speculative_num_steps=args.draft_tokens - 1,
            )
        engine = Engine(
            model_path=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            random_seed=args.seed,
            mem_fraction_static=args.mem_fraction_static,
            context_length=args.context_length,
            max_running_requests=args.max_running_requests,
            tp_size=1,
            schedule_policy="fcfs",
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
            enable_return_hidden_states=False,
            enable_metrics=True,
            attention_backend="triton",
            sampling_backend="pytorch",
            log_level="error",
            **engine_kwargs,
        )
        try:
            engine.flush_cache()
            if args.warmup_rows:
                count = args.warmup_rows
                engine.generate(
                    input_ids=prompt_ids[:count], sampling_params=sampling_params, rid=rids[:count]
                )
                engine.flush_cache()
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = engine.generate(
                input_ids=prompt_ids, sampling_params=sampling_params, rid=rids
            )
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - started
            if args.plugin:
                # The worker writes the head timing of the finished run on a cache flush.
                head_timing_path.unlink(missing_ok=True)
                engine.flush_cache()
                head_timing = json.loads(head_timing_path.read_text(encoding="utf-8"))
        finally:
            engine.shutdown()

    predictions = []
    for row, output in zip(rows, outputs):
        token_ids = [int(token_id) for token_id in output["output_ids"]]
        if token_ids and token_ids[-1] == tokenizer.eos_token_id:
            token_ids = token_ids[:-1]
        predictions.append(
            {
                "task_id": row["task_id"],
                "prediction": tokenizer.decode(token_ids, skip_special_tokens=True),
                "generated_token_ids": token_ids,
                "output_tokens": len(token_ids),
            }
        )
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    output_tokens = sum(row["output_tokens"] for row in predictions)
    stats = {
        "config": vars(args),
        "requests": len(predictions),
        "output_tokens": output_tokens,
        "wall_seconds": wall_seconds,
        "tokens_per_second": output_tokens / wall_seconds,
    }
    if head_timing is not None:
        head_seconds = sum(head_timing["head_cuda_ms"].values()) / 1000.0
        stats.update(head_timing)
        stats["head_share_percent"] = 100.0 * head_seconds / wall_seconds
    write_json(output_dir / "stats.json", stats)
    print(json.dumps({key: value for key, value in stats.items() if key != "config"}), flush=True)


if __name__ == "__main__":
    main()
