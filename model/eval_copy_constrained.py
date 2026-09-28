#!/usr/bin/env python3
"""
Decode with the constrained copy plugin:
  - trigger decides GEN vs COPY
  - length head proposes one copy span length
  - source matcher supplies aligned source starts
  - frozen base LLM verifies each proposal and accepts the longest safe prefix

Example:
python model/eval_copy_constrained.py \
  --model-path MODEL_PATH \
  --plugin outputs/checkpoints/random_sequence_plugin/plugin.pt \
  --input-jsonl data/random_sequence/test.jsonl \
  --output outputs/predictions/random_sequence_predictions.jsonl \
  --cuda-visible-devices 0 \
  --disable-thinking \
  --debug-trace-out outputs/traces/random_sequence_predictions.trace.jsonl
"""

from __future__ import annotations

import os
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv
from model.prompt_utils import encode_editor_prefix

apply_cuda_visible_devices_from_argv(sys.argv)

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from model.matcher import SourceMatcher
    from model.model import CopyConstraintModel, TRIGGER_TO_ID
    from model.sampling_utils import select_next_token, validate_sampling_args
except ModuleNotFoundError:
    from matcher import SourceMatcher
    from model import CopyConstraintModel, TRIGGER_TO_ID
    from sampling_utils import select_next_token, validate_sampling_args

MARGIN_ADJUSTMENTS = ("learned", "scale", "shift")


@dataclass
class VerifiedProposal:
    start: int
    proposed_length: int
    accepted_length: int
    margin_sum: float
    logprob_sum: float
    proposal_ids: list[int] | None = None
    verifier_token_trace: list[dict] | None = None


def _decode_token(tokenizer, token_id: int) -> str:
    if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
        return "<EOS>"
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _topk_debug(logits: torch.Tensor, tokenizer, k: int) -> list[dict]:
    if k <= 0:
        return []
    vals, idx = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
    out: list[dict] = []
    for logit, tok_id in zip(vals.tolist(), idx.tolist()):
        out.append(
            {
                "id": int(tok_id),
                "token": _decode_token(tokenizer, int(tok_id)),
                "logit": float(logit),
            }
        )
    return out


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _token_rank(logits: torch.Tensor, token_id: int) -> int:
    token_logit = logits[int(token_id)]
    return int((logits > token_logit).sum().item()) + 1


def _entropy_from_logits(logits: torch.Tensor) -> float:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return float((-(probs * log_probs)).sum().item())


def _energy_from_logits(logits: torch.Tensor) -> float:
    return float(torch.logsumexp(logits, dim=-1).item())


def _reference_next_token(target_ids: list[int], out_len: int, eos_id: int) -> dict | None:
    if target_ids is None:
        return None
    if out_len < len(target_ids):
        token_id = int(target_ids[out_len])
        kind = "target_token"
    elif out_len == len(target_ids):
        token_id = int(eos_id)
        kind = "eos_after_target"
    else:
        return None
    return {
        "kind": kind,
        "id": token_id,
    }


def _summarize_reference_next(
    logits: torch.Tensor,
    reference_next: dict | None,
    tokenizer,
    lm_topk: list[dict],
) -> dict | None:
    if reference_next is None:
        return None

    token_id = int(reference_next["id"])
    token_logit = float(logits[token_id].item())
    top1_logit = float(lm_topk[0]["logit"]) if lm_topk else None
    return {
        "kind": reference_next["kind"],
        "id": token_id,
        "token": _decode_token(tokenizer, token_id),
        "logit": token_logit,
        "rank": _token_rank(logits, token_id),
        "in_lm_topk": any(int(item["id"]) == token_id for item in lm_topk),
        "margin_vs_top1": None if top1_logit is None else float(token_logit - top1_logit),
    }


def _best_non_source_token(
    logits: torch.Tensor,
    source_token_ids: set[int],
    *,
    search_topk: int = 32,
) -> tuple[int, float]:
    k = min(max(search_topk, len(source_token_ids) + 2), logits.shape[-1])
    top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
    for logit, tok_id in zip(top_vals.tolist(), top_idx.tolist()):
        tok_id = int(tok_id)
        if tok_id not in source_token_ids:
            return tok_id, float(logit)

    keep_mask = torch.ones_like(logits, dtype=torch.bool)
    if source_token_ids:
        keep_mask[list(sorted(source_token_ids))] = False
    non_source_logits = logits.masked_fill(~keep_mask, float("-inf"))
    best_val, best_idx = torch.max(non_source_logits, dim=-1)
    return int(best_idx.item()), float(best_val.item())


_WS_RE = re.compile(r"\s+")


def _normalize_variant_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _source_continuation_text(tokenizer, source_ids: list[int], start: int, max_chars: int = 64) -> str:
    if not (0 <= start < len(source_ids)):
        return ""
    pieces: list[str] = []
    char_count = 0
    for tok_id in source_ids[start : min(len(source_ids), start + 16)]:
        piece = _decode_token(tokenizer, int(tok_id))
        pieces.append(piece)
        char_count += len(piece)
        if char_count >= max_chars:
            break
    return "".join(pieces)[:max_chars]


def _is_source_compatible_variant(token_text: str, source_text: str) -> bool:
    variant_norm = _normalize_variant_text(token_text)
    source_norm = _normalize_variant_text(source_text)
    if not variant_norm or not source_norm:
        return False
    if len(variant_norm) > 32:
        return False
    return source_norm.startswith(variant_norm) or variant_norm.startswith(source_norm[: len(variant_norm)])


def _source_compatible_topk_variants(
    logits: torch.Tensor,
    tokenizer,
    source_ids: list[int],
    candidate_starts: list[int],
    existing_token_ids: set[int],
    *,
    topk: int = 16,
) -> dict[int, dict]:
    if tokenizer is None or not candidate_starts or topk <= 0:
        return {}

    vals, idx = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
    top_tokens = [(int(tok_id), float(logit)) for logit, tok_id in zip(vals.tolist(), idx.tolist())]

    source_texts = []
    for start in candidate_starts:
        text = _source_continuation_text(tokenizer, source_ids, int(start))
        if text:
            source_texts.append((int(start), text))
    if not source_texts:
        return {}

    variants: dict[int, dict] = {}
    for tok_id, logit in top_tokens:
        if tok_id in existing_token_ids:
            continue
        token_text = _decode_token(tokenizer, tok_id)
        for start, source_text in source_texts:
            if _is_source_compatible_variant(token_text, source_text):
                variants[tok_id] = {
                    "id": tok_id,
                    "token": token_text,
                    "logit": logit,
                    "source_start": start,
                    "source_text": source_text,
                    "kind": "text_variant",
                }
                break
    return variants


def _summarize_source_support(
    logits: torch.Tensor,
    source_ids: list[int],
    candidate_starts: list[int],
    tokenizer,
    eos_id: int,
    *,
    reference_next: dict | None = None,
) -> dict | None:
    if not candidate_starts:
        return None

    next_token_counts: Counter[int] = Counter()
    for start in candidate_starts:
        if 0 <= start < len(source_ids):
            next_token_counts[int(source_ids[start])] += 1
    if not next_token_counts:
        return None

    token_ids = sorted(next_token_counts)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    source_logits = logits.index_select(0, token_tensor)
    best_source_idx = int(torch.argmax(source_logits).item())
    best_source_id = int(token_ids[best_source_idx])
    best_source_logit = float(source_logits[best_source_idx].item())
    best_non_source_id, best_non_source_logit = _best_non_source_token(logits, set(token_ids))

    follow_lse = float(torch.logsumexp(source_logits, dim=0).item())
    all_lse = _energy_from_logits(logits)
    if follow_lse >= all_lse:
        exit_lse = float("-inf")
    else:
        exit_lse = float(all_lse + math.log1p(-math.exp(follow_lse - all_lse)))

    modal_token_id, modal_count = max(
        next_token_counts.items(),
        key=lambda item: (item[1], float(logits[int(item[0])].item())),
    )

    eos_logit = float(logits[eos_id].item())
    next_tokens = []
    for token_id, count in sorted(
        next_token_counts.items(),
        key=lambda item: (-item[1], -float(logits[int(item[0])].item()), int(item[0])),
    ):
        next_tokens.append(
            {
                "id": int(token_id),
                "token": _decode_token(tokenizer, int(token_id)),
                "count": int(count),
                "logit": float(logits[int(token_id)].item()),
            }
        )

    summary = {
        "candidate_start_count": len(candidate_starts),
        "unique_next_token_count": len(next_token_counts),
        "candidate_agreement": float(modal_count / max(len(candidate_starts), 1)),
        "modal_next_token_id": int(modal_token_id),
        "modal_next_token": _decode_token(tokenizer, int(modal_token_id)),
        "modal_count": int(modal_count),
        "next_tokens": next_tokens,
        "best_source_id": best_source_id,
        "best_source_token": _decode_token(tokenizer, best_source_id),
        "best_source_logit": best_source_logit,
        "best_source_rank": _token_rank(logits, best_source_id),
        "best_non_source_id": int(best_non_source_id),
        "best_non_source_token": _decode_token(tokenizer, int(best_non_source_id)),
        "best_non_source_logit": best_non_source_logit,
        "best_source_margin_vs_best_non_source": float(best_source_logit - best_non_source_logit),
        "follow_logsumexp": follow_lse,
        "exit_logsumexp": _finite_or_none(exit_lse),
        "follow_vs_exit": None if not math.isfinite(exit_lse) else float(follow_lse - exit_lse),
        "eos_logit": eos_logit,
        "eos_rank": _token_rank(logits, eos_id),
        "best_source_margin_vs_eos": float(best_source_logit - eos_logit),
    }
    if reference_next is not None:
        ref_id = int(reference_next["id"])
        summary["reference_in_source"] = bool(ref_id in next_token_counts)
    return summary


def _fixed_margin_rule(
    margins: torch.Tensor,
    rule: str,
    *,
    scale: float,
    shift: float,
    grounding_length: int,
    match_threshold: int,
) -> torch.Tensor:
    """Fixed alternative to the learned bias, applied once the source match is long enough."""
    if grounding_length < match_threshold:
        return margins
    if rule == "scale":
        return margins * scale
    if rule == "shift":
        return margins + shift
    raise ValueError(f"Unknown margin adjustment: {rule}")


def _rerank_fallback_logits_with_source_bias(
    model: CopyConstraintModel,
    hidden_state: torch.Tensor,
    logits: torch.Tensor,
    source_ids: list[int],
    candidate_starts: list[int],
    device: torch.device,
    *,
    tokenizer=None,
    debug_topk: int = 0,
    include_text_variants: bool = False,
    margin_adjustment: str = "learned",
    heuristic_margin_scale: float = 1.0,
    heuristic_margin_shift: float = 0.0,
    grounding_length: int = 0,
    heuristic_match_threshold: int = 0,
) -> tuple[torch.Tensor, dict | None]:
    if not candidate_starts:
        return logits, None

    candidate_token_ids: list[int] = []
    seen_token_ids: set[int] = set()
    token_kinds: dict[int, str] = {}
    canonical_token_by_start: dict[int, int] = {}
    for start in candidate_starts:
        if 0 <= start < len(source_ids):
            tok_id = int(source_ids[start])
            canonical_token_by_start[int(start)] = tok_id
            if tok_id not in seen_token_ids:
                candidate_token_ids.append(tok_id)
                seen_token_ids.add(tok_id)
                token_kinds[tok_id] = "exact_source"
    variant_debug: dict[int, dict] = {}
    # Keep debug tracing observational: normal decoding only reranks exact
    # source-continuation tokens, so merely passing a tokenizer for trace output
    # must not add text-compatible token variants and change the generated text.
    if include_text_variants and tokenizer is not None:
        variant_debug = _source_compatible_topk_variants(
            logits,
            tokenizer,
            source_ids,
            candidate_starts,
            seen_token_ids,
            topk=max(debug_topk, 16),
        )
        for tok_id in variant_debug:
            if tok_id not in seen_token_ids:
                candidate_token_ids.append(tok_id)
                seen_token_ids.add(tok_id)
                token_kinds[tok_id] = "text_variant"
    if not candidate_token_ids:
        return logits, None

    token_tensor = torch.tensor(candidate_token_ids, dtype=torch.long, device=device)
    src_logits = logits.index_select(0, token_tensor)

    top_vals, top_idx = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1)
    top1_id = int(top_idx[0].item())
    top1_logit = float(top_vals[0].item())
    second_best_logit = float(top_vals[1].item()) if top_vals.shape[0] > 1 else float("-inf")

    alt_logits = torch.full_like(src_logits, top1_logit)
    if top_vals.shape[0] > 1:
        alt_logits = torch.where(token_tensor.eq(top1_id), torch.full_like(src_logits, second_best_logit), alt_logits)

    raw_margins = src_logits - alt_logits
    bias_hidden = hidden_state
    if bias_hidden.ndim == 1:
        bias_hidden = bias_hidden.unsqueeze(0)
    bias_hidden = bias_hidden.expand(token_tensor.shape[0], -1)
    if margin_adjustment == "learned":
        source_bias = model.source_token_bias(bias_hidden, raw_margins).float()
        adjusted_margins = raw_margins.float() + source_bias
    else:
        adjusted_margins = _fixed_margin_rule(
            raw_margins.float(),
            margin_adjustment,
            scale=heuristic_margin_scale,
            shift=heuristic_margin_shift,
            grounding_length=grounding_length,
            match_threshold=heuristic_match_threshold,
        )
        source_bias = adjusted_margins - raw_margins.float()

    adjusted_logits = alt_logits + adjusted_margins
    # Text-compatible tokenizer variants should not bypass the verifier for the
    # canonical source token they correspond to.
    variant_cap_tolerance = 0.0
    token_index = {int(tok_id): idx for idx, tok_id in enumerate(candidate_token_ids)}
    variant_caps: dict[int, dict] = {}
    for tok_id, info in variant_debug.items():
        var_idx = token_index.get(int(tok_id))
        canonical_id = canonical_token_by_start.get(int(info.get("source_start", -1)))
        canonical_idx = token_index.get(int(canonical_id)) if canonical_id is not None else None
        if var_idx is None or canonical_idx is None:
            continue
        before = adjusted_logits[var_idx]
        cap = adjusted_logits[canonical_idx] + variant_cap_tolerance
        adjusted_logits[var_idx] = torch.minimum(before, cap)
        variant_caps[int(tok_id)] = {
            "canonical_id": int(canonical_id),
            "canonical_token": _decode_token(tokenizer, int(canonical_id)) if tokenizer is not None else "",
            "canonical_adjusted_logit": float(adjusted_logits[canonical_idx].item()),
            "cap_tolerance": float(variant_cap_tolerance),
            "pre_cap_logit": float(before.item()),
            "post_cap_logit": float(adjusted_logits[var_idx].item()),
            "cap_applied": bool(adjusted_logits[var_idx].item() < before.item()),
        }

    adjusted_logits = adjusted_logits.to(logits.dtype)
    reranked_logits = logits.clone()
    reranked_logits.index_copy_(0, token_tensor, adjusted_logits)

    debug = None
    if tokenizer is not None and debug_topk > 0:
        debug = {
            "candidate_count": len(candidate_starts),
            "unique_token_count": len(candidate_token_ids),
            "raw_topk": _topk_debug(logits, tokenizer, debug_topk),
            "reranked_topk": _topk_debug(reranked_logits, tokenizer, debug_topk),
            "source_tokens": [],
            "variant_token_count": sum(1 for tok_id in candidate_token_ids if token_kinds.get(tok_id) == "text_variant"),
        }
        for tok_id, src_logit, alt_logit, raw_margin, bias, adjusted_logit in zip(
            candidate_token_ids,
            src_logits.tolist(),
            alt_logits.tolist(),
            raw_margins.tolist(),
            source_bias.tolist(),
            adjusted_logits.tolist(),
        ):
            debug["source_tokens"].append(
                {
                    "id": int(tok_id),
                    "token": _decode_token(tokenizer, int(tok_id)),
                    "kind": token_kinds.get(int(tok_id), "exact_source"),
                    "variant_match": variant_debug.get(int(tok_id)),
                    "variant_cap": variant_caps.get(int(tok_id)),
                    "src_logit": float(src_logit),
                    "alt_logit": float(alt_logit),
                    "raw_margin": float(raw_margin),
                    "source_bias": float(bias),
                    "reranked_logit": float(adjusted_logit),
                }
            )
    return reranked_logits, debug


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


def get_target_text(row: dict) -> str:
    target = row.get("gold_answer", row.get("edit_content", ""))
    if not isinstance(target, str):
        return ""
    return target


def _retokenize_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def get_source_ids(tokenizer, row: dict) -> list[int]:
    return _retokenize_ids(tokenizer, row["context"])


def get_target_ids(tokenizer, row: dict) -> list[int]:
    return _retokenize_ids(tokenizer, get_target_text(row))


def load_plugin_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def restore_plugin(model: CopyConstraintModel, ckpt: dict) -> str:
    state = ckpt.get("plugin_state_dict")
    if state is None:
        raise KeyError("Checkpoint is missing plugin_state_dict")
    model.load_plugin_state_dict(state)
    return ckpt.get("model_path", "")


def _percentile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((len(sorted_vals) - 1) * q))
    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _hist_bins(lengths: list[int], bin_edges: list[int] | None = None) -> list[dict]:
    if bin_edges is None:
        bin_edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    bins = []
    start = 1
    for edge in bin_edges:
        bins.append((start, edge))
        start = edge + 1
    bins.append((start, 10**9))

    out = []
    for lo, hi in bins:
        cnt = sum(1 for x in lengths if lo <= x <= hi)
        label = f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"
        out.append({"bin": label, "count": cnt})
    return out


def _exact_length_counts(lengths: list[int]) -> list[dict]:
    counts = Counter(lengths)
    return [{"length": int(length), "count": int(counts[length])} for length in sorted(counts)]


def _resolve_hist_plot_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return path
    return path.with_suffix(".png")


def _save_copy_length_plot(lengths: list[int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(lengths)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]

    fig_width = min(max(10.0, 0.18 * max(len(xs), 1)), 28.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    if xs:
        ax.bar(xs, ys, width=0.9, color="#2F855A", edgecolor="#1C4532", linewidth=0.4)
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
    else:
        ax.text(0.5, 0.5, "No accepted COPY operations", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Accepted Copy Length Frequency")
    ax.set_xlabel("Accepted Copy Length")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def verify_proposal(
    model: CopyConstraintModel,
    hidden_state: torch.Tensor,
    past_key_values,
    proposal_ids: list[int],
    device: torch.device,
    margin_floor: float = 0.0,
    *,
    margin_calibration: bool = True,
    margin_adjustment: str = "learned",
    heuristic_margin_scale: float = 1.0,
    heuristic_margin_shift: float = 0.0,
    grounding_length: int = 0,
    heuristic_match_threshold: int = 0,
    tokenizer=None,
    debug_topk: int = 0,
) -> tuple[int, float, float, list[dict] | None]:
    if not proposal_ids:
        return 0, float("-inf"), float("-inf"), None

    predictor_states, logits = model.proposal_predictor_states_and_logits(
        hidden_state,
        past_key_values,
        proposal_ids,
        device=device,
    )

    proposal_t = torch.tensor(proposal_ids, dtype=torch.long, device=device)
    src_logits = logits.gather(1, proposal_t.unsqueeze(1)).squeeze(1)

    top_vals, top_idx = torch.topk(logits, k=2, dim=-1)
    alt_logits = torch.where(top_idx[:, 0].eq(proposal_t), top_vals[:, 1], top_vals[:, 0])
    raw_margins = src_logits - alt_logits
    if margin_adjustment != "learned":
        margins = _fixed_margin_rule(
            raw_margins,
            margin_adjustment,
            scale=heuristic_margin_scale,
            shift=heuristic_margin_shift,
            grounding_length=grounding_length,
            match_threshold=heuristic_match_threshold,
        )
        source_bias = margins - raw_margins
    elif margin_calibration:
        source_bias = model.source_token_bias(predictor_states, raw_margins).float()
        margins = raw_margins + source_bias
    else:
        source_bias = torch.zeros_like(raw_margins, dtype=torch.float32)
        margins = raw_margins + source_bias

    accepted = 0
    for margin in margins.tolist():
        if margin > margin_floor:
            accepted += 1
        else:
            break

    record_token_trace = tokenizer is not None and debug_topk >= 0

    if accepted == 0:
        token_trace = None
        if record_token_trace:
            token_trace = []
            for pos, tok_id in enumerate(proposal_ids):
                top_idx_list = top_idx[pos].tolist()
                top_vals_list = top_vals[pos].tolist()
                if top_idx_list[0] == tok_id and len(top_idx_list) > 1:
                    alt_id = int(top_idx_list[1])
                    alt_logit = float(top_vals_list[1])
                else:
                    alt_id = int(top_idx_list[0])
                    alt_logit = float(top_vals_list[0])
                token_trace.append(
                    {
                        "offset": pos,
                        "proposal_id": int(tok_id),
                        "proposal_token": _decode_token(tokenizer, int(tok_id)),
                        "src_logit": float(src_logits[pos].item()),
                        "alt_id": alt_id,
                        "alt_token": _decode_token(tokenizer, alt_id),
                        "alt_logit": alt_logit,
                        "raw_margin": float(raw_margins[pos].item()),
                        "source_bias": float(source_bias[pos].item()),
                        "margin": float(margins[pos].item()),
                        "accepted": False,
                        "topk": _topk_debug(logits[pos], tokenizer, debug_topk),
                    }
                )
        return 0, float("-inf"), float("-inf"), token_trace

    log_probs = F.log_softmax(logits[:accepted], dim=-1)
    accepted_targets = proposal_t[:accepted]
    accepted_logprob = log_probs.gather(1, accepted_targets.unsqueeze(1)).sum().item()
    accepted_margin = margins[:accepted].sum().item()
    token_trace = None
    if record_token_trace:
        token_trace = []
        for pos, tok_id in enumerate(proposal_ids):
            top_idx_list = top_idx[pos].tolist()
            top_vals_list = top_vals[pos].tolist()
            if top_idx_list[0] == tok_id and len(top_idx_list) > 1:
                alt_id = int(top_idx_list[1])
                alt_logit = float(top_vals_list[1])
            else:
                alt_id = int(top_idx_list[0])
                alt_logit = float(top_vals_list[0])
            token_trace.append(
                {
                    "offset": pos,
                    "proposal_id": int(tok_id),
                    "proposal_token": _decode_token(tokenizer, int(tok_id)),
                    "src_logit": float(src_logits[pos].item()),
                    "alt_id": alt_id,
                    "alt_token": _decode_token(tokenizer, alt_id),
                    "alt_logit": alt_logit,
                    "raw_margin": float(raw_margins[pos].item()),
                        "source_bias": float(source_bias[pos].item()),
                        "margin": float(margins[pos].item()),
                        "accepted": bool(pos < accepted),
                        "topk": _topk_debug(logits[pos], tokenizer, debug_topk),
                    }
                )
    return accepted, float(accepted_margin), float(accepted_logprob), token_trace


def choose_best_proposal(
    model: CopyConstraintModel,
    hidden_state: torch.Tensor,
    past_key_values,
    source_ids: list[int],
    candidate_starts: list[int],
    proposed_length: int,
    device: torch.device,
    margin_floor: float,
    *,
    margin_calibration: bool = True,
    margin_adjustment: str = "learned",
    heuristic_margin_scale: float = 1.0,
    heuristic_margin_shift: float = 0.0,
    grounding_length: int = 0,
    heuristic_match_threshold: int = 0,
    tokenizer=None,
    debug_topk: int = 0,
) -> tuple[VerifiedProposal | None, list[dict] | None]:
    best: VerifiedProposal | None = None
    candidate_debug: list[dict] | None = [] if tokenizer is not None and debug_topk > 0 else None
    for start in candidate_starts:
        proposal_ids = source_ids[start : start + proposed_length]
        if not proposal_ids:
            continue
        accepted_length, margin_sum, logprob_sum, token_trace = verify_proposal(
            model,
            hidden_state,
            past_key_values,
            proposal_ids,
            device=device,
            margin_floor=margin_floor,
            margin_calibration=margin_calibration,
            margin_adjustment=margin_adjustment,
            heuristic_margin_scale=heuristic_margin_scale,
            heuristic_margin_shift=heuristic_margin_shift,
            grounding_length=grounding_length,
            heuristic_match_threshold=heuristic_match_threshold,
            tokenizer=tokenizer,
            debug_topk=debug_topk,
        )
        cand = VerifiedProposal(
            start=start,
            proposed_length=len(proposal_ids),
            accepted_length=accepted_length,
            margin_sum=margin_sum,
            logprob_sum=logprob_sum,
            proposal_ids=list(proposal_ids),
            verifier_token_trace=token_trace,
        )
        if candidate_debug is not None:
            candidate_debug.append(
                {
                    "start": cand.start,
                    "proposed_length": cand.proposed_length,
                    "accepted_length": cand.accepted_length,
                    "margin_sum": _finite_or_none(cand.margin_sum),
                    "logprob_sum": _finite_or_none(cand.logprob_sum),
                    "proposal_ids": list(proposal_ids),
                    "proposal_text": model.tokenizer.decode(proposal_ids, skip_special_tokens=False),
                }
            )
        if best is None:
            best = cand
            continue

        cand_key = (cand.accepted_length, cand.margin_sum, cand.logprob_sum, -cand.start)
        best_key = (best.accepted_length, best.margin_sum, best.logprob_sum, -best.start)
        if cand_key > best_key:
            best = cand
    return best, candidate_debug


@torch.no_grad()
def decode_one(
    model: CopyConstraintModel,
    row: dict,
    device: torch.device,
    max_steps: int,
    max_output_tokens: int,
    max_candidate_starts: int,
    fallback_lm_token_topk: int,
    verifier_margin_floor: float,
    min_match_length: int,
    disable_thinking: bool,
    calibrated_fallback: bool,
    fixed_copy_length: int,
    margin_calibration: bool,
    margin_adjustment: str = "learned",
    heuristic_margin_scale: float = 1.0,
    heuristic_margin_shift: float = 0.0,
    heuristic_match_threshold: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    generator: torch.Generator | None = None,
    *,
    debug_trace: bool = False,
    debug_topk: int = 5,
) -> tuple[str, list[int], int, str, dict, dict | None, list[int]]:
    tokenizer = model.tokenizer
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    source_ids = get_source_ids(tokenizer, row)
    target_ids = get_target_ids(tokenizer, row)
    matcher = SourceMatcher(
        source_ids,
        tokenizer=tokenizer,
    )

    prefix_ids = encode_editor_prefix(
        tokenizer,
        row["context"],
        row["instruction"],
        device,
        disable_thinking=disable_thinking,
    )
    hidden_state, past = model.decode_last_hidden(prefix_ids, use_cache=True)

    sampling = {"temperature": temperature, "top_p": top_p, "min_p": min_p}
    margin_rule = {
        "margin_adjustment": margin_adjustment,
        "heuristic_margin_scale": heuristic_margin_scale,
        "heuristic_margin_shift": heuristic_margin_shift,
        "heuristic_match_threshold": heuristic_match_threshold,
    }
    out_ids: list[int] = []
    copy_lengths: list[int] = []
    debug = {
        "copy_trigger_count": 0,
        "copy_accept_count": 0,
        "copy_fallback_to_gen_count": 0,
        "copy_proposed_token_count": 0,
        "copy_accepted_token_count": 0,
    }
    trace_steps: list[dict] | None = [] if debug_trace else None

    stop_reason = "max_steps"
    step = 0
    for step in range(1, max_steps + 1):
        if len(out_ids) >= max_output_tokens:
            stop_reason = "max_output_tokens"
            break

        features = matcher.current_features()
        vocab_logits = model.vocab_logits(hidden_state)
        step_logits = vocab_logits[0]
        lm_topk = _topk_debug(step_logits, tokenizer, debug_topk) if debug_trace else []
        reference_next = _reference_next_token(target_ids, len(out_ids), eos_id)
        aligned_candidate_starts: list[int] = []
        aligned_source_support = None
        if features.longest_match_length >= min_match_length:
            aligned_candidate_starts = matcher.candidate_next_starts(
                max_candidates=max_candidate_starts,
                longest_only=True,
            )
            if debug_trace:
                aligned_source_support = _summarize_source_support(
                    step_logits,
                    source_ids,
                    aligned_candidate_starts,
                    tokenizer,
                    eos_id,
                    reference_next=reference_next,
                )
        step_trace = None
        if trace_steps is not None:
            step_trace = {
                "step": step,
                "out_len_before": len(out_ids),
                "match_length": int(features.longest_match_length),
                "delta_match_length": int(features.delta_match_length),
                "lm_topk": lm_topk,
                "lm_stats": {
                    "entropy": _entropy_from_logits(step_logits),
                    "energy": _energy_from_logits(step_logits),
                    "top1_minus_top2": None if len(lm_topk) < 2 else float(lm_topk[0]["logit"] - lm_topk[1]["logit"]),
                },
                "reference_next": _summarize_reference_next(step_logits, reference_next, tokenizer, lm_topk),
                "aligned_candidate_starts": list(aligned_candidate_starts),
                "aligned_source_support": aligned_source_support,
                "decision": None,
                "decision_reason": None,
                "trigger_logits": None,
                "proposed_length": None,
                "candidate_starts": None,
                "candidate_source_support": None,
                "candidate_proposals": None,
                "selected_candidate": None,
                "emitted_ids": [],
                "emitted_text": "",
                "out_len_after": None,
            }
        if features.longest_match_length < min_match_length:
            gen_tok = select_next_token(step_logits, generator=generator, **sampling)
            # When source evidence is weak, allow the base LM to stop immediately.
            # Once matcher evidence is strong enough, do not let raw EOS bypass
            # proposal verification / source-biased reranking.
            if gen_tok == eos_id:
                stop_reason = "eos"
                if step_trace is not None:
                    step_trace["decision"] = "STOP"
                    step_trace["decision_reason"] = "eos_low_match"
                    step_trace["out_len_after"] = len(out_ids)
                    trace_steps.append(step_trace)
                break
            out_ids.append(gen_tok)
            matcher.append(gen_tok)
            gen_tok_t = torch.tensor([[gen_tok]], dtype=torch.long, device=device)
            hidden_state, past = model.decode_last_hidden(gen_tok_t, past_key_values=past, use_cache=True)
            if step_trace is not None:
                step_trace["decision"] = "GEN"
                step_trace["decision_reason"] = "below_min_match_length"
                step_trace["emitted_ids"] = [int(gen_tok)]
                step_trace["emitted_text"] = tokenizer.decode([gen_tok], skip_special_tokens=False)
                step_trace["out_len_after"] = len(out_ids)
                trace_steps.append(step_trace)
            continue

        trig_logits = model.trigger_logits(
            hidden_state,
            features.longest_match_length,
        ).float()
        trigger_id = int(trig_logits.argmax(dim=-1).item())
        trigger_mode = "COPY" if trigger_id == TRIGGER_TO_ID["COPY"] else "GEN"
        if step_trace is not None:
            step_trace["trigger_logits"] = trig_logits[0].tolist()

        if trigger_mode == "COPY":
            debug["copy_trigger_count"] += 1
            candidate_starts: list[int] = []
            if fixed_copy_length > 0:
                proposed_length = fixed_copy_length
            else:
                proposed_length = int(
                    model.predict_copy_length(
                        hidden_state,
                        features.longest_match_length,
                    )
                    .view(-1)[0]
                    .item()
                )
            if step_trace is not None:
                step_trace["proposed_length"] = proposed_length
            if proposed_length > 0:
                candidate_starts = list(aligned_candidate_starts)
                if not candidate_starts and fallback_lm_token_topk > 0:
                    topk = min(fallback_lm_token_topk, vocab_logits.shape[-1])
                    fallback_tokens = torch.topk(vocab_logits, k=topk, dim=-1).indices[0].tolist()
                    candidate_starts = matcher.fallback_candidate_starts(
                        fallback_tokens,
                        max_candidates=max_candidate_starts,
                    )
                if step_trace is not None:
                    step_trace["candidate_starts"] = list(candidate_starts)
                    step_trace["candidate_source_support"] = _summarize_source_support(
                        step_logits,
                        source_ids,
                        candidate_starts,
                        tokenizer,
                        eos_id,
                        reference_next=reference_next,
                    )

                best, candidate_debug = choose_best_proposal(
                    model,
                    hidden_state,
                    past,
                    source_ids,
                    candidate_starts,
                    proposed_length=proposed_length,
                    device=device,
                    margin_floor=verifier_margin_floor,
                    margin_calibration=margin_calibration,
                    tokenizer=tokenizer if debug_trace else None,
                    debug_topk=debug_topk if debug_trace else 0,
                    grounding_length=features.longest_match_length,
                    **margin_rule,
                )
                if step_trace is not None:
                    step_trace["candidate_proposals"] = candidate_debug
                if best is not None and best.accepted_length > 0:
                    remaining_tokens = max_output_tokens - len(out_ids)
                    accepted_length = min(best.accepted_length, remaining_tokens)
                    accepted_ids = source_ids[best.start : best.start + accepted_length]
                    if not accepted_ids:
                        stop_reason = "max_output_tokens"
                        if step_trace is not None:
                            step_trace["decision"] = "STOP"
                            step_trace["decision_reason"] = "max_output_tokens_after_copy_clip"
                            step_trace["out_len_after"] = len(out_ids)
                            trace_steps.append(step_trace)
                        break
                    copy_tensor = torch.tensor([accepted_ids], dtype=torch.long, device=device)
                    hidden_state, past = model.decode_last_hidden(
                        copy_tensor,
                        past_key_values=past,
                        use_cache=True,
                    )
                    out_ids.extend(accepted_ids)
                    matcher.extend(accepted_ids)
                    copy_lengths.append(len(accepted_ids))
                    debug["copy_accept_count"] += 1
                    debug["copy_proposed_token_count"] += best.proposed_length
                    debug["copy_accepted_token_count"] += len(accepted_ids)
                    if step_trace is not None:
                        step_trace["decision"] = "COPY"
                        step_trace["decision_reason"] = "copy_accepted"
                        step_trace["selected_candidate"] = {
                            "start": best.start,
                            "proposed_length": best.proposed_length,
                            "accepted_length": best.accepted_length,
                            "margin_sum": best.margin_sum,
                            "logprob_sum": best.logprob_sum,
                            "proposal_ids": best.proposal_ids,
                            "proposal_text": tokenizer.decode(best.proposal_ids or [], skip_special_tokens=False),
                            "accepted_ids": list(accepted_ids),
                            "accepted_text": tokenizer.decode(accepted_ids, skip_special_tokens=False),
                            "verifier_token_trace": best.verifier_token_trace,
                        }
                        step_trace["emitted_ids"] = list(accepted_ids)
                        step_trace["emitted_text"] = tokenizer.decode(accepted_ids, skip_special_tokens=False)
                        step_trace["out_len_after"] = len(out_ids)
                        trace_steps.append(step_trace)
                    continue

            debug["copy_fallback_to_gen_count"] += 1
            if step_trace is not None:
                step_trace["decision"] = "GEN"
                step_trace["decision_reason"] = "copy_fallback_to_gen"
            if calibrated_fallback:
                reranked_logits, fallback_rerank = _rerank_fallback_logits_with_source_bias(
                    model,
                    hidden_state,
                    step_logits,
                    source_ids,
                    candidate_starts,
                    device=device,
                    tokenizer=tokenizer if debug_trace else None,
                    debug_topk=debug_topk if debug_trace else 0,
                    grounding_length=features.longest_match_length,
                    **margin_rule,
                )
            else:
                reranked_logits = step_logits
                fallback_rerank = None
                if step_trace is not None:
                    step_trace["fallback_source_rerank"] = {
                        "enabled": False,
                        "reason": "calibrated_fallback_disabled",
                    }
            gen_tok = select_next_token(reranked_logits, generator=generator, **sampling)
            if step_trace is not None and fallback_rerank is not None:
                step_trace["fallback_source_rerank"] = fallback_rerank
        else:
            gen_tok = select_next_token(step_logits, generator=generator, **sampling)

        if gen_tok == eos_id:
            stop_reason = "eos"
            if step_trace is not None:
                step_trace["decision"] = "STOP"
                if step_trace["decision_reason"] == "copy_fallback_to_gen":
                    step_trace["decision_reason"] = "copy_fallback_selected_eos"
                elif step_trace["decision_reason"] is None:
                    step_trace["decision_reason"] = "selected_eos"
                step_trace["emitted_ids"] = []
                step_trace["emitted_text"] = ""
                step_trace["out_len_after"] = len(out_ids)
                trace_steps.append(step_trace)
            break

        out_ids.append(gen_tok)
        matcher.append(gen_tok)
        gen_tok_t = torch.tensor([[gen_tok]], dtype=torch.long, device=device)
        hidden_state, past = model.decode_last_hidden(gen_tok_t, past_key_values=past, use_cache=True)
        if step_trace is not None:
            if step_trace["decision"] is None:
                step_trace["decision"] = "GEN"
                step_trace["decision_reason"] = "trigger_gen"
            step_trace["emitted_ids"] = [int(gen_tok)]
            step_trace["emitted_text"] = tokenizer.decode([gen_tok], skip_special_tokens=False)
            step_trace["out_len_after"] = len(out_ids)
            trace_steps.append(step_trace)

    trace_record = None
    if trace_steps is not None:
        trace_record = {
            "task_id": row.get("task_id"),
            "instruction": row.get("instruction"),
            "gold_answer": row.get("gold_answer", row.get("edit_content")),
            "gold_target_ids": list(target_ids),
            "output_ids": list(out_ids),
            "prediction": tokenizer.decode(out_ids, skip_special_tokens=False),
            "stop_reason": stop_reason,
            "steps": trace_steps,
        }

    return tokenizer.decode(out_ids, skip_special_tokens=False), copy_lengths, step, stop_reason, debug, trace_record, out_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode with a trigger+length constrained copy plugin.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter path to load on top of --model-path.")
    parser.add_argument("--plugin", required=True, help="Path to plugin.pt")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output", default="outputs/predictions/copy_constraint_predictions.jsonl")
    parser.add_argument("--max-steps", type=int, default=4096, help="Maximum decision steps per sample")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--copy-stats-out", default="outputs/predictions/copy_constraint_copy_stats.json")
    parser.add_argument("--copy-hist-out", default="outputs/predictions/copy_constraint_copy_hist.png")
    parser.add_argument("--max-candidate-starts", type=int, default=8)
    parser.add_argument("--fallback-lm-token-topk", type=int, default=4)
    parser.add_argument("--verifier-margin-floor", type=float, default=0.0)
    parser.add_argument("--min-match-length", type=int, default=5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for GEN and fallback steps; copy verification stays deterministic.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="Sample i is decoded with seed + i.")
    parser.add_argument(
        "--fixed-copy-length",
        type=int,
        default=0,
        help="If >0, bypass the length head and use this copy length whenever COPY is triggered.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '1'. Applied before torch import.",
    )
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--calibrated-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When COPY falls back to GEN, apply the learned source-token bias to fallback logits. "
            "Use --no-calibrated-fallback for the ablation where fallback uses frozen logits directly."
        ),
    )
    parser.add_argument(
        "--margin-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When verifying copy proposals, add the learned source-token bias to token-level margins. "
            "Use --no-margin-calibration to verify with raw margins only."
        ),
    )
    parser.add_argument(
        "--margin-adjustment",
        choices=MARGIN_ADJUSTMENTS,
        default="learned",
        help=(
            "Margin correction used by the verifier and the calibrated fallback: the learned source bias, "
            "or a fixed rule that scales or shifts the raw margin (the learned bias head is then unused)."
        ),
    )
    parser.add_argument("--heuristic-margin-scale", type=float, default=1.0)
    parser.add_argument("--heuristic-margin-shift", type=float, default=0.0)
    parser.add_argument(
        "--heuristic-match-threshold",
        type=int,
        default=0,
        help="Apply the fixed rule only when the source match length is at least this value.",
    )
    parser.add_argument("--debug-trace-out", default="", help="Optional JSONL path for per-step decode traces.")
    parser.add_argument("--debug-topk", type=int, default=5, help="Top-k logits to store in debug traces.")
    args = parser.parse_args()
    if args.fixed_copy_length < 0:
        parser.error("--fixed-copy-length must be non-negative")
    if args.heuristic_margin_scale <= 0:
        parser.error("--heuristic-margin-scale must be positive")
    try:
        validate_sampling_args(args.temperature, args.top_p, args.min_p)
    except ValueError as exc:
        parser.error(str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    ckpt = load_plugin_checkpoint(Path(args.plugin))

    disable_thinking = ckpt.get("disable_thinking", False) if args.disable_thinking is None else args.disable_thinking
    adapter_path = args.adapter_path or ckpt.get("adapter_path") or None

    model = CopyConstraintModel(
        args.model_path,
        adapter_path=adapter_path,
        freeze_base=True,
        torch_dtype=dtype,
        dropout=ckpt.get("dropout", 0.1),
        source_bias_scale=ckpt.get("source_bias_scale", 6.0),
    ).to(device)
    restore_plugin(model, ckpt)
    model.eval()

    model_name = args.model_name or f"{Path(args.model_path).name}+copy_constraint_plugin"

    print(
        f"loaded copy-constrained model from {args.model_path} "
        f"(adapter={adapter_path or 'none'}, dtype={dtype}, disable_thinking={disable_thinking}, "
        f"margin_calibration={args.margin_calibration}, margin_adjustment={args.margin_adjustment}, "
        f"temperature={args.temperature}, top_p={args.top_p}, min_p={args.min_p}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
    )

    rows = load_rows(Path(args.input_jsonl), limit=args.max_samples)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_copy_lengths: list[int] = []
    steps_list: list[int] = []
    stop_reason_counts: Counter[str] = Counter()
    copy_trigger_count = 0
    copy_accept_count = 0
    copy_fallback_to_gen_count = 0
    copy_proposed_token_count = 0
    copy_accepted_token_count = 0
    output_token_count = 0
    decode_seconds_total = 0.0
    sample_tokens_per_second: list[float] = []
    trace_path = Path(args.debug_trace_out) if args.debug_trace_out else None
    trace_fh = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = trace_path.open("w", encoding="utf-8")

    with out_path.open("w", encoding="utf-8") as f:
        try:
            for sample_idx, row in enumerate(tqdm(rows, desc="decoding", unit="sample")):
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + sample_idx)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                sample_start = time.perf_counter()
                pred, copy_lengths, steps, stop_reason, debug, trace_record, output_ids = decode_one(
                    model,
                    row,
                    device=device,
                    max_steps=args.max_steps,
                    max_output_tokens=args.max_output_tokens,
                    max_candidate_starts=args.max_candidate_starts,
                    fallback_lm_token_topk=args.fallback_lm_token_topk,
                    verifier_margin_floor=args.verifier_margin_floor,
                    min_match_length=args.min_match_length,
                    disable_thinking=disable_thinking,
                    calibrated_fallback=args.calibrated_fallback,
                    fixed_copy_length=args.fixed_copy_length,
                    margin_calibration=args.margin_calibration,
                    margin_adjustment=args.margin_adjustment,
                    heuristic_margin_scale=args.heuristic_margin_scale,
                    heuristic_margin_shift=args.heuristic_margin_shift,
                    heuristic_match_threshold=args.heuristic_match_threshold,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    generator=generator,
                    debug_trace=trace_fh is not None,
                    debug_topk=args.debug_topk,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                sample_decode_seconds = time.perf_counter() - sample_start
                sample_output_tokens = len(output_ids)

                all_copy_lengths.extend(copy_lengths)
                steps_list.append(steps)
                stop_reason_counts[stop_reason] += 1
                copy_trigger_count += debug["copy_trigger_count"]
                copy_accept_count += debug["copy_accept_count"]
                copy_fallback_to_gen_count += debug["copy_fallback_to_gen_count"]
                copy_proposed_token_count += debug["copy_proposed_token_count"]
                copy_accepted_token_count += debug["copy_accepted_token_count"]
                output_token_count += sample_output_tokens
                decode_seconds_total += sample_decode_seconds
                if sample_decode_seconds > 0:
                    sample_tokens_per_second.append(sample_output_tokens / sample_decode_seconds)

                rec = {
                    "task_id": row.get("task_id"),
                    "case_id": row.get("case_id"),
                    "edit_type": row.get("edit_type"),
                    "task_type": row.get("task_type"),
                    "source_split": row.get("source_split"),
                    "source_dataset": row.get("source_dataset"),
                    "instruction": row.get("instruction"),
                    "context": row.get("context"),
                    "gold_answer": row.get("gold_answer", row.get("edit_content")),
                    "model_name": model_name,
                    "prediction": pred,
                    "model_edited_answer": pred,
                    "generated_token_ids": output_ids,
                    "decode_seconds": sample_decode_seconds,
                    "stop_reason": stop_reason,
                    "copy_lengths": copy_lengths,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if trace_fh is not None and trace_record is not None:
                    trace_fh.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
        finally:
            if trace_fh is not None:
                trace_fh.close()

    sorted_lens = sorted(all_copy_lengths)
    copy_stats = {
        "num_samples": len(rows),
        "max_steps": args.max_steps,
        "max_output_tokens": args.max_output_tokens,
        "max_candidate_starts": args.max_candidate_starts,
        "fallback_lm_token_topk": args.fallback_lm_token_topk,
        "calibrated_fallback": bool(args.calibrated_fallback),
        "margin_calibration": bool(args.margin_calibration),
        "margin_adjustment": args.margin_adjustment,
        "heuristic_margin_scale": args.heuristic_margin_scale,
        "heuristic_margin_shift": args.heuristic_margin_shift,
        "heuristic_match_threshold": args.heuristic_match_threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
        "fixed_copy_length": int(args.fixed_copy_length),
        "verifier_margin_floor": args.verifier_margin_floor,
        "min_match_length": args.min_match_length,
        "source_bias_scale": float(model.source_bias_scale),
        "stop_reason_counts": dict(stop_reason_counts),
        "avg_steps": mean(steps_list) if steps_list else 0.0,
        "copy_trigger_count": copy_trigger_count,
        "copy_accept_count": copy_accept_count,
        "copy_fallback_to_gen_count": copy_fallback_to_gen_count,
        "copy_accept_rate_given_trigger": (copy_accept_count / copy_trigger_count) if copy_trigger_count else 0.0,
        "copy_token_accept_ratio": (copy_accepted_token_count / copy_proposed_token_count) if copy_proposed_token_count else 0.0,
        "output_token_count": output_token_count,
        "decode_seconds_total": decode_seconds_total,
        "tokens_per_second_overall": (output_token_count / decode_seconds_total) if decode_seconds_total > 0 else 0.0,
        "tokens_per_second_mean_per_sample": mean(sample_tokens_per_second) if sample_tokens_per_second else 0.0,
        "tokens_per_second_median_per_sample": median(sample_tokens_per_second) if sample_tokens_per_second else 0.0,
        "copy_op_count": len(all_copy_lengths),
        "copy_len_mean": mean(all_copy_lengths) if all_copy_lengths else 0.0,
        "copy_len_median": median(all_copy_lengths) if all_copy_lengths else 0.0,
        "copy_len_p90": _percentile(sorted_lens, 0.9),
        "copy_len_p95": _percentile(sorted_lens, 0.95),
        "copy_len_max": max(all_copy_lengths) if all_copy_lengths else 0,
        "copy_len_counts": _exact_length_counts(all_copy_lengths),
        "copy_len_hist": _hist_bins(all_copy_lengths),
    }

    stats_path = Path(args.copy_stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(copy_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hist_path = _resolve_hist_plot_path(args.copy_hist_out)
    _save_copy_length_plot(all_copy_lengths, hist_path)

    print(f"saved predictions -> {out_path}")
    print(f"saved copy stats -> {stats_path}")
    print(f"saved copy plot  -> {hist_path}")
    if trace_path is not None:
        print(f"saved debug trace -> {trace_path}")


if __name__ == "__main__":
    main()
