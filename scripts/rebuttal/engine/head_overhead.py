#!/usr/bin/env python3
"""Decode-time overhead of the trained CopyGen heads inside vLLM or SGLang.

A logits processor runs the heads on zero inputs with the shapes of one decode
step and multiplies their outputs by zero, so every mode generates the same
tokens. ``bias`` runs the source-bias head on --candidate-items rows per step;
``all3`` also runs the trigger and length heads; ``noop`` registers the same
processor without running any head. Prompts are stratified by prompt length
(--samples-per-domain per domain); for every prompt and repeat the modes run
back to back in random order at batch size 1 with a fixed output length.

The overhead of a mode is the mean over (prompt, repeat) pairs of
100 * (request wall / noop request wall - 1), with a prompt-level bootstrap CI.

Example:
  python -m scripts.rebuttal.engine.head_overhead --engine vllm \
    --model-path MODEL_PATH \
    --oracle contract=data/contract/test.jsonl --plugin contract=CONTRACT_PLUGIN_PT \
    --oracle literature=data/literature/test.jsonl --plugin literature=LITERATURE_PLUGIN_PT \
    --oracle code=data/code/test.jsonl --plugin code=CODE_PLUGIN_PT \
    --attention-backend TRITON_ATTN --output outputs/rebuttal/engine/head_overhead_vllm.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from statistics import fmean

from transformers import AutoTokenizer

from scripts.rebuttal.engine.common import (
    load_jsonl,
    parse_domain_paths,
    render_editor_prompt,
    write_json,
)

MODES = ("noop", "bias", "all3")


def stratified_sample(lengths: list[int], count: int, rng: random.Random) -> list[int]:
    """One random index from each of ``count`` equal-size prompt-length strata."""
    ordered = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    selected = []
    for stratum in range(count):
        begin = stratum * len(ordered) // count
        end = (stratum + 1) * len(ordered) // count
        selected.append(rng.choice(ordered[begin:end]))
    return selected


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], resamples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = [fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(resamples)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def overhead_summary(records: list[dict], mode: str, resamples: int, seed: int) -> dict:
    walls = {
        (r["domain"], r["task_id"], r["repeat"], r["mode"]): r["wall_seconds"] for r in records
    }
    per_prompt: dict[tuple[str, str], list[float]] = {}
    values = []
    for (domain, task_id, repeat, record_mode), noop_wall in sorted(walls.items()):
        if record_mode != "noop":
            continue
        value = 100.0 * (walls[(domain, task_id, repeat, mode)] / noop_wall - 1.0)
        values.append(value)
        per_prompt.setdefault((domain, task_id), []).append(value)
    prompt_means = [fmean(items) for _, items in sorted(per_prompt.items())]
    return {
        "pairs": len(values),
        "prompts": len(prompt_means),
        "wall_overhead_percent": fmean(values),
        "wall_overhead_percent_ci95": bootstrap_mean_ci(prompt_means, resamples, seed),
    }


class VllmRunner:
    def __init__(self, args, plugins: dict[str, str]) -> None:
        os.environ["COPYGEN_VLLM_CONFIG"] = json.dumps(
            {"plugins": plugins, "candidate_items": args.candidate_items}
        )
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        kwargs = dict(
            model=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_fraction,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            enable_prefix_caching=False,
            enforce_eager=False,
            disable_log_stats=False,
            logits_processors=["scripts.rebuttal.engine.vllm_copygen:HeadOverheadProcessor"],
        )
        if args.attention_backend != "auto":
            kwargs["attention_config"] = {"backend": args.attention_backend}
        self.llm = LLM(**kwargs)

    def run(self, sample: dict, domain: str, mode: str, max_tokens: int):
        params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            seed=0,
            extra_args={"copygen_domain": domain, "copygen_mode": mode},
        )
        started = time.perf_counter()
        output = self.llm.generate([sample["prompt"]], params, use_tqdm=False)[0]
        wall_seconds = time.perf_counter() - started
        return list(output.outputs[0].token_ids), wall_seconds

    def close(self) -> None:
        pass


class SglangRunner:
    def __init__(self, args, plugins: dict[str, str]) -> None:
        os.environ["COPYGEN_SGLANG_HEADS"] = json.dumps(
            {"plugins": plugins, "candidate_items": args.candidate_items, "max_batch_rows": 1}
        )
        from sglang.srt.entrypoints.engine import Engine

        from scripts.rebuttal.engine.sglang_copygen import HeadOverheadProcessor

        self.processor = HeadOverheadProcessor.to_str()
        self.engine = Engine(
            model_path=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            random_seed=args.seed,
            mem_fraction_static=args.gpu_memory_fraction,
            context_length=args.max_model_len,
            max_running_requests=1,
            tp_size=1,
            schedule_policy="fcfs",
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
            enable_custom_logit_processor=True,
            enable_metrics=True,
            log_level="error",
        )

    def run(self, sample: dict, domain: str, mode: str, max_tokens: int):
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "min_p": 0.0,
            "sampling_seed": 0,
            "max_new_tokens": max_tokens,
            "ignore_eos": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": True,
            "custom_params": {"copygen_domain": domain, "copygen_mode": mode},
        }
        started = time.perf_counter()
        output = self.engine.generate(
            input_ids=sample["prompt_ids"],
            sampling_params=sampling_params,
            custom_logit_processor=self.processor,
        )
        wall_seconds = time.perf_counter() - started
        return list(output["output_ids"]), wall_seconds

    def close(self) -> None:
        self.engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--oracle", action="append", required=True, help="DOMAIN=PATH (repeatable)")
    parser.add_argument(
        "--plugin", action="append", required=True, help="DOMAIN=PLUGIN_PT (repeatable)"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--candidate-items", type=int, default=8)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.80)
    parser.add_argument("--attention-backend", default="auto", help="vLLM only")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    oracles = parse_domain_paths(args.oracle)
    plugins = {
        domain: str(Path(path).resolve())
        for domain, path in parse_domain_paths(args.plugin).items()
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rng = random.Random(args.seed)
    selected: dict[str, list[dict]] = {}
    for domain, path in oracles.items():
        prepared = []
        for row in load_jsonl(path):
            prompt = render_editor_prompt(
                tokenizer,
                row["context"],
                row["instruction"],
                disable_thinking=args.disable_thinking,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prepared.append({"task_id": row["task_id"], "prompt": prompt, "prompt_ids": prompt_ids})
        indices = stratified_sample(
            [len(item["prompt_ids"]) for item in prepared], args.samples_per_domain, rng
        )
        rng.shuffle(indices)
        selected[domain] = [prepared[index] for index in indices]

    runner = (VllmRunner if args.engine == "vllm" else SglangRunner)(args, plugins)
    records = []
    try:
        for domain, samples in selected.items():
            for mode in args.modes:
                runner.run(samples[0], domain, mode, args.warmup_tokens)
        for domain, samples in selected.items():
            for sample in samples:
                for repeat in range(args.repeats):
                    tokens_by_mode = {}
                    for mode in rng.sample(args.modes, len(args.modes)):
                        token_ids, wall_seconds = runner.run(sample, domain, mode, args.max_tokens)
                        tokens_by_mode[mode] = token_ids
                        records.append(
                            {
                                "domain": domain,
                                "task_id": sample["task_id"],
                                "repeat": repeat,
                                "mode": mode,
                                "prompt_tokens": len(sample["prompt_ids"]),
                                "wall_seconds": wall_seconds,
                                "generated_token_ids": token_ids,
                            }
                        )
                    if any(tokens != tokens_by_mode["noop"] for tokens in tokens_by_mode.values()):
                        raise RuntimeError(
                            f"{domain}/{sample['task_id']}: head modes changed the generated tokens"
                        )
                    print(
                        f"{domain} {sample['task_id']} repeat={repeat} "
                        + " ".join(
                            f"{r['mode']}={r['wall_seconds']:.3f}s"
                            for r in records[-len(args.modes) :]
                        ),
                        flush=True,
                    )
    finally:
        runner.close()

    summary = {
        mode: overhead_summary(records, mode, args.bootstrap_resamples, args.seed)
        for mode in args.modes
        if mode != "noop"
    }
    write_json(args.output, {"config": vars(args), "summary": summary, "records": records})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
