#!/usr/bin/env python3
"""CopyGen decoding on vLLM target execution, verifying every candidate source start.

The client runs the CopyGen control loop of model/eval_copy_constrained.py; vLLM
(ProbeWorker) supplies the target logits, hidden states and head outputs. Every
candidate is verified token by token, candidates are ranked by (accepted length,
margin sum, log-prob sum, -start), and a rejected COPY falls back to the
calibrated rerank. Rows are
shuffled with --seed and decoded in wavefronts of --wavefront-size.

Example:
  python -m scripts.rebuttal.engine.vllm_full_decode \
    --model-path MODEL_PATH \
    --oracle contract=data/contract/test.jsonl --plugin contract=CONTRACT_PLUGIN_PT \
    --oracle literature=data/literature/test.jsonl --plugin literature=LITERATURE_PLUGIN_PT \
    --output-dir outputs/rebuttal/engine/vllm_full_decode
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.rebuttal.engine.common import (
    SourceMatcher,
    load_jsonl,
    parse_domain_paths,
    render_editor_prompt,
    write_json,
    write_jsonl,
)

MAX_CANDIDATE_STARTS = 8
FALLBACK_LM_TOKEN_TOPK = 4
MIN_MATCH_LENGTH = 5
MARGIN_FLOOR = 0.0
PROBE_WINDOW_TOKENS = 128


@dataclass
class DecodeState:
    index: int
    domain: str
    task_id: str
    prompt_ids: list[int]
    source_ids: list[int]
    matcher: SourceMatcher
    output_ids: list[int] = field(default_factory=list)
    copy_lengths: list[int] = field(default_factory=list)
    steps: int = 0
    stop_reason: str | None = None

    @property
    def key(self) -> str:
        return f"{self.domain}:{self.task_id}"

    @property
    def prefix_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.output_ids]


@dataclass
class CandidatePlan:
    start: int
    proposal_ids: list[int]
    windows: list[tuple[int, int]]


def unique_first_tokens(source_ids, starts) -> list[int]:
    return list(dict.fromkeys(int(source_ids[start]) for start in starts))


def build_plans(source_ids, starts, proposed_length: int) -> list[CandidatePlan]:
    """One proposal per start, split into windows of at most PROBE_WINDOW_TOKENS."""
    plans = []
    for start in starts:
        proposal = [int(token_id) for token_id in source_ids[start : start + proposed_length]]
        if not proposal:
            continue
        windows = [
            (begin, min(begin + PROBE_WINDOW_TOKENS, len(proposal)))
            for begin in range(0, len(proposal), PROBE_WINDOW_TOKENS)
        ]
        plans.append(CandidatePlan(int(start), proposal, windows))
    return plans


def verify_plan(plan: CandidatePlan, rows: list[tuple[int, float, float]]):
    """Accept the longest prefix whose adjusted margins all exceed the floor."""
    rows = sorted(rows)
    if [offset for offset, _, _ in rows] != list(range(len(plan.proposal_ids))):
        raise RuntimeError(f"incomplete evidence for candidate start {plan.start}")
    accepted = 0
    for _, adjusted, _ in rows:
        if adjusted > MARGIN_FLOOR:
            accepted += 1
        else:
            break
    if accepted:
        margin_sum = float(sum(adjusted for _, adjusted, _ in rows[:accepted]))
        logprob_sum = float(sum(logprob for _, _, logprob in rows[:accepted]))
    else:
        margin_sum = logprob_sum = float("-inf")
    return {
        "start": plan.start,
        "proposal_ids": plan.proposal_ids,
        "accepted_length": accepted,
        "rank_key": (accepted, margin_sum, logprob_sum, -plan.start),
    }


def greedy_top_scores(probe: dict, sampled_token_id: int) -> list[tuple[int, float]]:
    """Top-9 target scores, with an exact top-logit tie resolved toward vLLM's sample."""
    top_ids = [int(token_id) for token_id in probe["top_ids"]]
    top_logits = [float(value) for value in probe["top_logits"]]
    if sampled_token_id != top_ids[0]:
        index = top_ids.index(sampled_token_id)
        if top_logits[index] != top_logits[0]:
            raise RuntimeError("vLLM greedy token has a lower logit than the traced top-1")
        top_ids[0], top_ids[index] = top_ids[index], top_ids[0]
        top_logits[0], top_logits[index] = top_logits[index], top_logits[0]
    return list(zip(top_ids, top_logits))


def calibrated_fallback(top_scores, candidate_token_ids, fallback: dict) -> int:
    """Rerank the source first tokens with their learned bias and pick greedily."""
    evidence = {
        int(token_id): (float(alternative), float(adjusted))
        for token_id, alternative, adjusted in zip(
            fallback["token_ids"], fallback["alternative_logits"], fallback["adjusted_margins"]
        )
    }
    reranked = dict(top_scores)
    for token_id in candidate_token_ids:
        alternative, adjusted = evidence[token_id]
        reranked[token_id] = alternative + adjusted
    return max(reranked.items(), key=lambda item: (item[1], -item[0]))[0]


def emit(state: DecodeState, token_id: int, eos_id: int) -> None:
    state.steps += 1
    if token_id == eos_id:
        state.stop_reason = "eos"
    else:
        state.output_ids.append(int(token_id))
        state.matcher.append(int(token_id))


def probe_params(SamplingParams, extra: dict, skip_prefix_cache: bool = False):
    return SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        seed=0,
        extra_args=extra,
        skip_reading_prefix_cache=True if skip_prefix_cache else None,
    )


def run_probes(llm, prompts, params, drain):
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    return outputs, drain(), wall_seconds


def decode_wavefront(llm, SamplingParams, drain, states, eos_id, args) -> float:
    """Advance all states with one state-probe batch and one candidate batch per step."""
    wall_seconds = 0.0
    while True:
        active = []
        for state in states:
            if state.stop_reason is not None:
                continue
            if len(state.output_ids) >= args.max_output_tokens:
                state.stop_reason = "max_output_tokens"
            elif state.steps >= args.max_steps:
                state.stop_reason = "max_steps"
            else:
                active.append(state)
        if not active:
            return wall_seconds

        prompts, params, keys = [], [], []
        for state in active:
            match_length = int(state.matcher.current_features().longest_match_length)
            aligned = state.matcher.candidate_next_starts(
                max_candidates=MAX_CANDIDATE_STARTS, longest_only=True
            )
            key = f"{state.key}:s{state.steps}:state"
            keys.append(key)
            prompts.append({"prompt_token_ids": state.prefix_ids})
            params.append(
                probe_params(
                    SamplingParams,
                    {
                        "copygen_probe_kind": "state",
                        "copygen_probe_key": key,
                        "copygen_domain": state.domain,
                        "copygen_match_length": match_length,
                        "copygen_aligned_first_token_ids": unique_first_tokens(
                            state.source_ids, aligned
                        ),
                    },
                )
            )
        outputs, results, seconds = run_probes(llm, prompts, params, drain)
        wall_seconds += seconds
        probes = {result["probe_key"]: result for result in results}

        pending = []
        for state, output, key in zip(active, outputs, keys):
            probe = probes[key]
            sampled = int(output.outputs[0].token_ids[0])
            top_scores = greedy_top_scores(probe, sampled)
            match_length = int(state.matcher.current_features().longest_match_length)
            trigger = probe["trigger_logits"]
            if (
                match_length < MIN_MATCH_LENGTH
                or trigger is None
                or not float(trigger[1]) > float(trigger[0])
            ):
                emit(state, sampled, eos_id)
                continue
            starts = state.matcher.candidate_next_starts(
                max_candidates=MAX_CANDIDATE_STARTS, longest_only=True
            )
            if not starts:
                fallback_tokens = [token_id for token_id, _ in top_scores[:FALLBACK_LM_TOKEN_TOPK]]
                starts = state.matcher.fallback_candidate_starts(
                    fallback_tokens, max_candidates=MAX_CANDIDATE_STARTS
                )
            plans = build_plans(state.source_ids, starts, int(probe["predicted_length"]))
            pending.append((state, probe, top_scores, plans))
        if not pending:
            continue

        # Every window of every candidate of every pending state in one batch.
        prompts, params, owners = [], [], {}
        for state, _, _, plans in pending:
            prefix = state.prefix_ids
            for plan in plans:
                for begin, end in plan.windows:
                    key = f"{state.key}:s{state.steps}:c{plan.start}:w{begin}-{end}"
                    owners[key] = (state, plan)
                    # The last window token is predicted by the preceding position,
                    # so the prompt ends right before it.
                    prompts.append(
                        {"prompt_token_ids": [*prefix, *plan.proposal_ids[: max(0, end - 1)]]}
                    )
                    params.append(
                        probe_params(
                            SamplingParams,
                            {
                                "copygen_probe_kind": "candidate",
                                "copygen_probe_key": key,
                                "copygen_domain": state.domain,
                                "copygen_prefix_length": len(prefix),
                                "copygen_proposal_ids": plan.proposal_ids,
                                "copygen_window_start": begin,
                                "copygen_window_end": end,
                            },
                            skip_prefix_cache=True,
                        )
                    )
        evidence: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
        if prompts:
            _, results, seconds = run_probes(llm, prompts, params, drain)
            wall_seconds += seconds
            for result in results:
                state, plan = owners[result["probe_key"]]
                rows = evidence.setdefault((state.key, plan.start), [])
                for offset, (adjusted, logprob) in enumerate(
                    zip(result["adjusted_margins"], result["logprobs"]),
                    start=result["offset_start"],
                ):
                    rows.append((offset, float(adjusted), float(logprob)))

        for state, probe, top_scores, plans in pending:
            verified = [
                verify_plan(plan, evidence.get((state.key, plan.start), [])) for plan in plans
            ]
            best = max(verified, key=lambda item: item["rank_key"]) if verified else None
            if best is not None and best["accepted_length"] > 0:
                length = min(
                    best["accepted_length"], args.max_output_tokens - len(state.output_ids)
                )
                accepted_ids = best["proposal_ids"][:length]
                state.steps += 1
                if not accepted_ids:
                    state.stop_reason = "max_output_tokens"
                else:
                    state.output_ids.extend(accepted_ids)
                    state.matcher.extend(accepted_ids)
                    state.copy_lengths.append(len(accepted_ids))
                continue
            if verified:
                candidate_ids = unique_first_tokens(
                    state.source_ids, [item["start"] for item in verified]
                )
                selected = calibrated_fallback(top_scores, candidate_ids, probe["fallback"])
            else:
                selected = top_scores[0][0]
            emit(state, selected, eos_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--oracle", action="append", required=True, help="DOMAIN=PATH (repeatable)")
    parser.add_argument(
        "--plugin", action="append", required=True, help="DOMAIN=PLUGIN_PT (repeatable)"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Writes DOMAIN.jsonl per domain and stats.json"
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--wavefront-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    oracles = parse_domain_paths(args.oracle)
    plugins = parse_domain_paths(args.plugin)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = []
    for domain, path in oracles.items():
        for row in load_jsonl(path):
            prompt = render_editor_prompt(
                tokenizer,
                row["context"],
                row["instruction"],
                disable_thinking=args.disable_thinking,
            )
            samples.append(
                {
                    "index": len(samples),
                    "domain": domain,
                    "task_id": row["task_id"],
                    "prompt_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
                    "source_ids": tokenizer(row["context"], add_special_tokens=False)["input_ids"],
                }
            )

    os.environ["COPYGEN_VLLM_CONFIG"] = json.dumps(
        {
            "plugins": {domain: str(Path(plugins[domain]).resolve()) for domain in oracles},
            "min_match_length": MIN_MATCH_LENGTH,
            "fallback_lm_token_topk": FALLBACK_LM_TOKEN_TOPK,
            "state_topn": MAX_CANDIDATE_STARTS + 1,
        }
    )
    from vllm import LLM, SamplingParams

    from scripts.rebuttal.engine.vllm_copygen import drain_probe_results

    llm_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=False,
        disable_log_stats=False,
        worker_cls="scripts.rebuttal.engine.vllm_copygen.ProbeWorker",
        async_scheduling=False,
    )
    if args.attention_backend != "auto":
        llm_kwargs["attention_config"] = {"backend": args.attention_backend}
    llm = LLM(**llm_kwargs)

    ordered = list(samples)
    random.Random(args.seed).shuffle(ordered)
    finished = []
    wall_seconds = 0.0
    for wave_start in range(0, len(ordered), args.wavefront_size):
        states = [
            DecodeState(
                index=sample["index"],
                domain=sample["domain"],
                task_id=sample["task_id"],
                prompt_ids=sample["prompt_ids"],
                source_ids=sample["source_ids"],
                matcher=SourceMatcher(sample["source_ids"], tokenizer=tokenizer),
            )
            for sample in ordered[wave_start : wave_start + args.wavefront_size]
        ]
        wall_seconds += decode_wavefront(
            llm, SamplingParams, drain_probe_results, states, tokenizer.eos_token_id, args
        )
        finished.extend(states)
        print(f"decoded {len(finished)}/{len(ordered)} rows", flush=True)

    finished.sort(key=lambda state: state.index)
    output_dir = Path(args.output_dir)
    for domain in oracles:
        rows = [
            {
                "task_id": state.task_id,
                "prediction": tokenizer.decode(state.output_ids, skip_special_tokens=True),
                "generated_token_ids": state.output_ids,
                "copy_lengths": state.copy_lengths,
                "stop_reason": state.stop_reason,
            }
            for state in finished
            if state.domain == domain
        ]
        write_jsonl(output_dir / f"{domain}.jsonl", rows)
    write_json(
        output_dir / "stats.json",
        {
            "config": vars(args),
            "rows": len(finished),
            "output_tokens": sum(len(state.output_ids) for state in finished),
            "copied_tokens": sum(sum(state.copy_lengths) for state in finished),
            "probe_wall_seconds": wall_seconds,
        },
    )


if __name__ == "__main__":
    main()
