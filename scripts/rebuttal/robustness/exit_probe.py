#!/usr/bin/env python3
"""Copy-exit probes: does CopyGen change the sampling distribution where copying should stop?

boundary      Teacher-force each gold answer (Contract, Literature, Code) and compare the frozen LM's
              sampler-filtered next-token distribution with CopyGen's decision at that state.
code-exit     Teacher-force the Code gold span inside a fixed answer layout and compare at the first
              token of the closing fence.
continuation  Also teacher-force the closing fence, then sample free continuations with the frozen LM
              (base) and with CopyGen, using the same random stream for both.

Example:
python -m scripts.rebuttal.robustness.exit_probe boundary \
  --model-path /path/to/Qwen3-8B \
  --plugin contract=/path/to/contract/plugin.pt \
  --plugin literature=/path/to/literature/plugin.pt \
  --plugin code=/path/to/code/plugin.pt \
  --output-dir outputs/exit_probe/boundary \
  --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv

apply_cuda_visible_devices_from_argv(sys.argv)

import torch

from model.eval_copy_constrained import (
    _rerank_fallback_logits_with_source_bias,
    choose_best_proposal,
    get_source_ids,
    get_target_ids,
    load_plugin_checkpoint,
    restore_plugin,
)
from model.matcher import SourceMatcher
from model.model import CopyConstraintModel, TRIGGER_TO_ID
from model.prompt_utils import encode_editor_prefix
from model.sampling_utils import filter_sampling_logits, select_next_token


MIN_MATCH_LENGTH = 5
MAX_CANDIDATE_STARTS = 8
FALLBACK_LM_TOKEN_TOPK = 4
SAMPLERS = {
    "top_p0p9": {"temperature": 0.7, "top_p": 0.9, "min_p": 0.0},
    "min_p0p05": {"temperature": 0.7, "top_p": 1.0, "min_p": 0.05},
}
CONTINUATION_SUFFIX = (
    "\nAfter the requested answer, continue with one short, varied sentence. "
    "Do not repeat the copied text."
)
CODE_ANSWER_HEADER = (
    "Explanation:\n"
    "The requested source span is reproduced below.\n\n"
    "Quoted block:\n"
    "````python\n"
)
CLOSING_FENCE = "````\n"
CODE_SYSTEM_PROMPT = (
    "You are a precise long-context coding assistant. First reproduce the "
    "requested source span exactly inside the required four-backtick Python "
    "fence. Immediately after the closing fence, write one concise, "
    "independently worded technical sentence."
)
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def write_outputs(output_dir: Path, records: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_model(model_path: str, ckpt: dict) -> CopyConstraintModel:
    model = CopyConstraintModel(
        model_path,
        adapter_path=None,
        freeze_base=True,
        torch_dtype=torch.bfloat16,
        dropout=ckpt.get("dropout", 0.1),
        source_bias_scale=ckpt.get("source_bias_scale", 6.0),
    ).to("cuda")
    restore_plugin(model, ckpt)
    model.eval()
    return model


def copygen_step(model, hidden_state, past, matcher: SourceMatcher, source_ids: list[int]):
    """CopyGen's decision at one state: (path, logits it samples from, copied ids)."""
    raw_logits = model.vocab_logits(hidden_state)[0]
    match_length = matcher.current_features().longest_match_length
    if match_length < MIN_MATCH_LENGTH:
        return "below_min_match", raw_logits, []
    trigger_logits = model.trigger_logits(hidden_state, match_length).float()
    if int(trigger_logits.argmax(dim=-1).item()) != TRIGGER_TO_ID["COPY"]:
        return "trigger_gen", raw_logits, []

    proposed_length = int(model.predict_copy_length(hidden_state, match_length).view(-1)[0].item())
    candidate_starts = matcher.candidate_next_starts(max_candidates=MAX_CANDIDATE_STARTS, longest_only=True)
    if not candidate_starts:
        fallback_tokens = torch.topk(raw_logits, k=FALLBACK_LM_TOKEN_TOPK).indices.tolist()
        candidate_starts = matcher.fallback_candidate_starts(fallback_tokens, max_candidates=MAX_CANDIDATE_STARTS)
    best, _ = choose_best_proposal(
        model,
        hidden_state,
        past,
        source_ids,
        candidate_starts,
        proposed_length=proposed_length,
        device=raw_logits.device,
        margin_floor=0.0,
        grounding_length=match_length,
    )
    if best is not None and best.accepted_length > 0:
        return "copy", raw_logits, source_ids[best.start : best.start + best.accepted_length]
    reranked_logits, _ = _rerank_fallback_logits_with_source_bias(
        model,
        hidden_state,
        raw_logits,
        source_ids,
        candidate_starts,
        raw_logits.device,
        grounding_length=match_length,
    )
    return "fallback", reranked_logits, []


def sampler_probs(logits: torch.Tensor, sampler: dict, drop_token: int | None = None):
    """Sampler-filtered next-token distribution and its support."""
    if drop_token is not None:
        logits = logits.clone()
        logits[drop_token] = float("-inf")
    filtered = filter_sampling_logits(logits, **sampler)
    return torch.softmax(filtered.float(), dim=-1), torch.isfinite(filtered)


def distribution_distance(p: torch.Tensor, q: torch.Tensor) -> dict:
    mixture = 0.5 * (p + q)
    kl = [(x[x > 0] * (x[x > 0].log() - mixture[x > 0].log())).sum() for x in (p, q)]
    return {
        "top1_changed": bool(torch.argmax(p) != torch.argmax(q)),
        "total_variation": float(0.5 * torch.abs(p - q).sum()),
        "jensen_shannon": float(0.5 * (kl[0] + kl[1])),
    }


def value_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered) if ordered else None,
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
        "max": ordered[-1] if ordered else None,
    }


def warm_state(model, token_ids: list[int]):
    return model.decode_last_hidden(torch.tensor([token_ids], dtype=torch.long, device="cuda"), use_cache=True)


@torch.no_grad()
def boundary_record(model, row: dict, domain: str, disable_thinking: bool) -> dict:
    tokenizer = model.tokenizer
    source_ids = get_source_ids(tokenizer, row)
    target_ids = get_target_ids(tokenizer, row)
    prefix_ids = encode_editor_prefix(
        tokenizer,
        row["context"],
        row["instruction"] + CONTINUATION_SUFFIX,
        torch.device("cuda"),
        disable_thinking=disable_thinking,
    )
    hidden_state, past = model.decode_last_hidden(prefix_ids, use_cache=True)
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device="cuda")
    hidden_state, past = model.decode_last_hidden(target_tensor, past_key_values=past, use_cache=True)

    matcher = SourceMatcher(source_ids, tokenizer=tokenizer)
    matcher.extend(target_ids)
    path, copygen_logits, copy_ids = copygen_step(model, hidden_state, past, matcher, source_ids)
    raw_logits = model.vocab_logits(hidden_state)[0]
    record = {
        "domain": domain,
        "task_id": row["task_id"],
        "match_length": int(matcher.current_features().longest_match_length),
        "path": path,
        "copy_length": len(copy_ids),
        "samplers": {},
    }
    if path == "copy":
        return record
    for name, sampler in SAMPLERS.items():
        metrics = {}
        for prefix, drop_token in (("", None), ("non_eos_", tokenizer.eos_token_id)):
            p, _ = sampler_probs(raw_logits, sampler, drop_token)
            q, _ = sampler_probs(copygen_logits, sampler, drop_token)
            p, q = p.double(), q.double()
            distance = distribution_distance(p / p.sum(), q / q.sum())
            metrics[prefix + "top1_changed"] = distance["top1_changed"]
            metrics[prefix + "total_variation"] = distance["total_variation"]
        record["samplers"][name] = metrics
    return record


def boundary_summary(records: list[dict]) -> dict:
    compared = [record for record in records if record["samplers"]]
    summary = {
        "states": len(records),
        "path_counts": dict(Counter(record["path"] for record in records)),
        "compared_states": len(compared),
        "samplers": {},
    }
    for name in SAMPLERS:
        metrics = [record["samplers"][name] for record in compared]
        summary["samplers"][name] = {
            "top1_changed": sum(m["top1_changed"] for m in metrics),
            "non_eos_top1_changed": sum(m["non_eos_top1_changed"] for m in metrics),
            "total_variation": value_summary([m["total_variation"] for m in metrics]),
            "non_eos_total_variation": value_summary([m["non_eos_total_variation"] for m in metrics]),
        }
    return summary


def run_boundary(args) -> None:
    plugins = dict(item.split("=", 1) for item in args.plugin)
    model = None
    records = []
    for domain, plugin_path in plugins.items():
        ckpt = load_plugin_checkpoint(Path(plugin_path))
        if model is None:
            model = load_model(args.model_path, ckpt)
        else:
            restore_plugin(model, ckpt)
            model.source_bias_scale = float(ckpt.get("source_bias_scale", 6.0))
        rows = load_rows(Path(args.data_dir) / domain / "test.jsonl", args.limit)
        for index, row in enumerate(rows, start=1):
            record = boundary_record(model, row, domain, bool(ckpt.get("disable_thinking", False)))
            records.append(record)
            print(f"[{domain} {index}/{len(rows)}] {row['task_id']} path={record['path']}", flush=True)
    write_outputs(Path(args.output_dir), records, boundary_summary(records))


def code_prompt_ids(tokenizer, row: dict) -> list[int]:
    user = (
        "Original task:\n\n"
        f"{row['instruction']}\n\n"
        "Source text:\n\n"
        f"{row['context']}\n\n"
        "Required output format:\n"
        "Explanation:\n"
        "<one brief explanation>\n\n"
        "Quoted block:\n"
        "````python\n"
        "<the exact requested source span>\n"
        "````\n"
        "<one new concise technical sentence about the role of the quoted block>\n\n"
        "Use exactly four backticks for both fences. The new sentence must begin "
        "on the line immediately after the closing fence. Do not add a marker, "
        "heading, or blank line between the closing fence and the new sentence."
    )
    messages = [{"role": "system", "content": CODE_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in (tokenizer.chat_template or ""):
        kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    return list(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def code_answer_ids(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    """Token ids of the answer up to the end of the gold span, and the closing fence tokens after it."""
    answer = CODE_ANSWER_HEADER + row["gold_evidence"]
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    with_fence_ids = list(tokenizer(answer + CLOSING_FENCE, add_special_tokens=False)["input_ids"])
    if with_fence_ids[: len(answer_ids)] != answer_ids:
        raise ValueError(f"{row['task_id']}: closing fence changes the tokenization of the answer")
    return answer_ids, with_fence_ids[len(answer_ids) :]


@torch.no_grad()
def code_exit_record(model, row: dict) -> dict:
    tokenizer = model.tokenizer
    source_ids = get_source_ids(tokenizer, row)
    answer_ids, fence_ids = code_answer_ids(tokenizer, row)
    hidden_state, past = warm_state(model, code_prompt_ids(tokenizer, row) + answer_ids)
    matcher = SourceMatcher(source_ids, tokenizer=tokenizer)
    matcher.extend(answer_ids)
    raw_logits = model.vocab_logits(hidden_state)[0]
    path, copygen_logits, copy_ids = copygen_step(model, hidden_state, past, matcher, source_ids)

    record = {
        "task_id": row["task_id"],
        "match_length": int(matcher.current_features().longest_match_length),
        "path": path,
        "fence_token_id": fence_ids[0],
        "copy_ids": copy_ids,
        "samplers": {},
    }
    for name, sampler in SAMPLERS.items():
        p, p_support = sampler_probs(raw_logits, sampler)
        if copy_ids:
            q = torch.zeros_like(p)
            q[copy_ids[0]] = 1.0
            q_support = q > 0
        else:
            q, q_support = sampler_probs(copygen_logits, sampler)
        metrics = distribution_distance(p, q)
        union = int((p_support | q_support).sum())
        metrics["support_jaccard"] = int((p_support & q_support).sum()) / union if union else 1.0
        record["samplers"][name] = metrics
    return record


def code_exit_summary(records: list[dict]) -> dict:
    copies = [record for record in records if record["path"] == "copy"]
    summary = {
        "states": len(records),
        "path_counts": dict(Counter(record["path"] for record in records)),
        "wrong_copy": sum(record["copy_ids"][0] != record["fence_token_id"] for record in copies),
        "samplers": {},
    }
    for name in SAMPLERS:
        summary["samplers"][name] = {}
        for subset, selected in (("all", records), ("non_copy", [r for r in records if r["path"] != "copy"])):
            metrics = [record["samplers"][name] for record in selected]
            summary["samplers"][name][subset] = {
                "states": len(metrics),
                "top1_changed": sum(m["top1_changed"] for m in metrics),
                "total_variation": value_summary([m["total_variation"] for m in metrics]),
                "jensen_shannon_mean": statistics.mean(m["jensen_shannon"] for m in metrics),
                "support_jaccard_mean": statistics.mean(m["support_jaccard"] for m in metrics),
                "support_jaccard_min": min(m["support_jaccard"] for m in metrics),
            }
    return summary


def run_code_exit(args) -> None:
    model = load_model(args.model_path, load_plugin_checkpoint(Path(args.plugin)))
    rows = load_rows(Path(args.input_jsonl), args.limit)
    records = []
    for index, row in enumerate(rows, start=1):
        records.append(code_exit_record(model, row))
        print(f"[{index}/{len(rows)}] {row['task_id']} path={records[-1]['path']}", flush=True)
    write_outputs(Path(args.output_dir), records, code_exit_summary(records))


def sample_seed(task_id: str, sampler_name: str, seed: int) -> int:
    """Per-task seed, shared by both arms so that they draw the same random stream."""
    digest = hashlib.sha256(f"{task_id}\0{sampler_name}\0{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@torch.no_grad()
def free_continuation(model, row: dict, arm: str, sampler: dict, seed: int, max_new_tokens: int) -> dict:
    tokenizer = model.tokenizer
    source_ids = get_source_ids(tokenizer, row)
    answer_ids, fence_ids = code_answer_ids(tokenizer, row)
    answer_ids = answer_ids + fence_ids
    hidden_state, past = warm_state(model, code_prompt_ids(tokenizer, row) + answer_ids)
    matcher = SourceMatcher(source_ids, tokenizer=tokenizer)
    matcher.extend(answer_ids)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    output_ids: list[int] = []
    trigger_calls = 0
    copy_count = 0
    stop_reason = "max_new_tokens"
    while len(output_ids) < max_new_tokens:
        if arm == "base":
            logits, copy_ids = model.vocab_logits(hidden_state)[0], []
        else:
            path, logits, copy_ids = copygen_step(model, hidden_state, past, matcher, source_ids)
            trigger_calls += path != "below_min_match"
        if copy_ids:
            copy_ids = copy_ids[: max_new_tokens - len(output_ids)]
            copy_count += 1
            new_ids = copy_ids
        else:
            token_id = select_next_token(logits, generator=generator, **sampler)
            if token_id == tokenizer.eos_token_id:
                stop_reason = "eos"
                break
            new_ids = [token_id]
        output_ids.extend(new_ids)
        matcher.extend(new_ids)
        new_tensor = torch.tensor([new_ids], dtype=torch.long, device="cuda")
        hidden_state, past = model.decode_last_hidden(new_tensor, past_key_values=past, use_cache=True)
    return {
        "generated_token_ids": output_ids,
        "prediction": tokenizer.decode(output_ids, skip_special_tokens=False),
        "stop_reason": stop_reason,
        "trigger_calls": trigger_calls,
        "copy_count": copy_count,
    }


def token_jaccard_distance(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in WORD_RE.findall(left)}
    right_tokens = {token.lower() for token in WORD_RE.findall(right)}
    union = left_tokens | right_tokens
    return 1.0 - len(left_tokens & right_tokens) / len(union) if union else 0.0


def diversity(texts_by_task: dict[str, list[str]]) -> dict:
    """Across-seed diversity per task, averaged over tasks."""
    unique, differ, jaccard = [], [], []
    for texts in texts_by_task.values():
        pairs = list(itertools.combinations(texts, 2))
        unique.append(len(set(texts)) / len(texts))
        differ.append(sum(a != b for a, b in pairs) / len(pairs))
        jaccard.append(statistics.mean(token_jaccard_distance(a, b) for a, b in pairs))
    return {
        "unique_rate": statistics.mean(unique),
        "pairwise_difference_rate": statistics.mean(differ),
        "token_jaccard_distance": statistics.mean(jaccard),
    }


def continuation_summary(records: list[dict], seeds: list[int]) -> dict:
    by_key = {(r["sampler"], r["seed"], r["arm"], r["task_id"]): r for r in records}
    task_ids = list(dict.fromkeys(r["task_id"] for r in records))
    summary = {}
    for name in SAMPLERS:
        pairs = [(by_key[(name, s, "base", t)], by_key[(name, s, "copygen", t)]) for s in seeds for t in task_ids]
        summary[name] = {
            "pairs": len(pairs),
            "token_identical_pairs": sum(b["generated_token_ids"] == c["generated_token_ids"] for b, c in pairs),
            "copygen_heads_active": sum(c["trigger_calls"] > 0 for _, c in pairs),
            "copygen_copies": sum(c["copy_count"] for _, c in pairs),
            "eos_stops": sum(r["stop_reason"] == "eos" for pair in pairs for r in pair),
        }
        if len(seeds) > 1:
            for arm in ("base", "copygen"):
                texts = {t: [by_key[(name, s, arm, t)]["prediction"].strip() for s in seeds] for t in task_ids}
                summary[name][arm] = diversity(texts)
    return summary


def run_continuation(args) -> None:
    model = load_model(args.model_path, load_plugin_checkpoint(Path(args.plugin)))
    rows = load_rows(Path(args.input_jsonl), args.limit)
    records = []
    for name, sampler in SAMPLERS.items():
        for seed in args.seeds:
            for arm in ("base", "copygen"):
                for row in rows:
                    decoded = free_continuation(
                        model, row, arm, sampler, sample_seed(row["task_id"], name, seed), args.max_new_tokens
                    )
                    records.append({"task_id": row["task_id"], "sampler": name, "seed": seed, "arm": arm, **decoded})
                print(f"[{name} seed={seed} {arm}] done", flush=True)
    write_outputs(Path(args.output_dir), records, continuation_summary(records, args.seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy-exit distribution and continuation probes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "boundary": (run_boundary, "next token after the teacher-forced gold answer"),
        "code-exit": (run_code_exit, "first closing-fence token after the Code gold span"),
        "continuation": (run_continuation, "free continuations after the closing fence"),
    }
    for name, (run, help_text) in commands.items():
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.set_defaults(run=run)
        subparser.add_argument("--model-path", required=True)
        subparser.add_argument("--output-dir", required=True)
        subparser.add_argument("--cuda-visible-devices", default=None)
        if name == "boundary":
            subparser.add_argument("--plugin", action="append", required=True, metavar="DOMAIN=PATH")
            subparser.add_argument("--data-dir", default="data")
            subparser.add_argument("--limit", type=int, default=0, help="rows per domain (0 = all)")
        else:
            subparser.add_argument("--plugin", required=True)
            subparser.add_argument("--input-jsonl", default="data/code/test.jsonl")
            subparser.add_argument("--limit", type=int, default=32 if name == "continuation" else 0)
        if name == "continuation":
            subparser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
            subparser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
