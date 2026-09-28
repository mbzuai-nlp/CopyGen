#!/usr/bin/env python3
"""
Train the speculative-style copy-constrained plugin:
  - frozen base LLM
  - trigger head: GEN vs COPY
  - length head: direct copy-span proposal length

Example:
python model/train_copy_constrained.py \
  --model-path MODEL_PATH \
  --train-jsonl data/random_sequence/train.jsonl \
  --save-dir outputs/checkpoints/random_sequence_plugin \
  --cuda-visible-devices 0 \
  --epochs 1 \
  --grad-accum-steps 16 \
  --disable-thinking
"""

from __future__ import annotations

import os
import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv

apply_cuda_visible_devices_from_argv(sys.argv)

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler

from model.prompt_utils import encode_editor_prefix

try:
    from model.labels import build_training_signals
    from model.model import (
        CopyConstraintModel,
        TRIGGER_TO_ID,
        asymmetric_length_regression_loss,
        trigger_nll_loss,
    )
except ModuleNotFoundError:
    from labels import build_training_signals
    from model import (
        CopyConstraintModel,
        TRIGGER_TO_ID,
        asymmetric_length_regression_loss,
        trigger_nll_loss,
    )

_REPORTED_FULL_FORWARD_OOM = False


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["__row_idx"] = row_idx
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_target_text(row: dict) -> str:
    target = row.get("gold_answer", row.get("edit_content", ""))
    if not isinstance(target, str):
        return ""
    return target


def _retokenize_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def get_source_and_target_ids(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    source_ids = _retokenize_ids(tokenizer, row["context"])
    target_ids = _retokenize_ids(tokenizer, get_target_text(row))
    return list(source_ids), list(target_ids)


def collect_target_prefix_states(
    model: CopyConstraintModel,
    prefix_ids: torch.Tensor,
    target_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Return one decoder state per target position:
      state[t] = hidden state after consuming prefix + target[:t]

    For frozen-base training this can be collected with one full teacher-forced
    forward over prefix+target, which is dramatically faster than replaying the
    target token by token with KV cache.
    """
    if not target_ids:
        hidden_size = model.config.hidden_size
        return torch.empty((0, hidden_size), dtype=model._base_parameter_dtype(), device=device)

    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
    full_input_ids = torch.cat([prefix_ids, target_tensor], dim=1)
    prefix_len = prefix_ids.shape[1]

    try:
        hidden_seq, _ = model.decode_hidden_sequence(full_input_ids, use_cache=False)
        state_tensor = hidden_seq[:, prefix_len - 1 : prefix_len - 1 + len(target_ids), :]
        return state_tensor.squeeze(0)
    except torch.OutOfMemoryError:
        global _REPORTED_FULL_FORWARD_OOM
        if not _REPORTED_FULL_FORWARD_OOM:
            print(
                "warning: full teacher-forced forward ran out of memory; "
                "falling back to incremental replay, which is much slower"
            )
            _REPORTED_FULL_FORWARD_OOM = True
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Fallback to incremental replay if a full forward does not fit.
        h_t, past = model.decode_last_hidden(prefix_ids, use_cache=True)
        states = [h_t]
        for tok in target_ids[:-1]:
            tok_t = torch.tensor([[tok]], dtype=torch.long, device=device)
            h_t, past = model.decode_last_hidden(tok_t, past_key_values=past, use_cache=True)
            states.append(h_t)
        return torch.cat(states, dim=0)


def format_row_debug(row: dict) -> str:
    return (
        f"row_idx={row.get('__row_idx', 'unknown')} "
        f"task_id={row.get('task_id', 'unknown')} "
        f"src_len={len(row.get('source_token_ids', []))} "
        f"tgt_len={len(row.get('target_token_ids', []))}"
    )


def build_bias_training_anchors(
    signals,
    source_ids: list[int],
    target_ids: list[int],
    min_match_length: int,
    max_spans: int,
    max_starts_per_anchor: int,
    stride_divisor: int,
) -> list[tuple[int, int, int]]:
    anchors: list[tuple[int, int, int]] = []
    t = 0
    stride_divisor = max(int(stride_divisor), 1)
    max_starts_per_anchor = max(int(max_starts_per_anchor), 1)
    while t < len(target_ids):
        has_match = signals.match_lengths[t] >= min_match_length
        longest = int(signals.longest_copy_lengths[t])
        starts = signals.best_copy_starts[t]
        if not has_match or longest <= 0 or not starts:
            t += 1
            continue

        valid_spans: list[tuple[int, int]] = []
        for start in starts[:max_starts_per_anchor]:
            start = int(start)
            max_gold_len = min(longest, len(target_ids) - t, len(source_ids) - start)
            if max_gold_len <= 0:
                continue
            valid_spans.append((start, max_gold_len))
        if not valid_spans:
            t += 1
            continue

        for start, max_gold_len in valid_spans:
            anchors.append((t, start, max_gold_len))
            if max_spans > 0 and len(anchors) >= max_spans:
                break
        if max_spans > 0 and len(anchors) >= max_spans:
            break

        best_gold_len = max(gold_len for _, gold_len in valid_spans)
        t += max(1, best_gold_len // stride_divisor)
    return anchors


def collect_proposal_prefix_state(
    model: CopyConstraintModel,
    prefix_ids: torch.Tensor,
    target_ids: list[int],
    target_pos: int,
    device: torch.device,
):
    if target_pos > 0:
        history_t = torch.tensor([target_ids[:target_pos]], dtype=torch.long, device=device)
        prefix_input = torch.cat([prefix_ids, history_t], dim=1)
    else:
        prefix_input = prefix_ids
    return model.decode_last_hidden(prefix_input, use_cache=True)


def proposal_source_margins(
    logits: torch.Tensor,
    proposal_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    proposal_t = torch.tensor(proposal_ids, dtype=torch.long, device=device)
    src_logits = logits.gather(1, proposal_t.unsqueeze(1)).squeeze(1)

    top_vals, top_idx = torch.topk(logits, k=2, dim=-1)
    alt_logits = torch.where(top_idx[:, 0].eq(proposal_t), top_vals[:, 1], top_vals[:, 0])
    return src_logits - alt_logits


def _decode_ids(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def build_text_boundary_accept_labels(
    tokenizer,
    source_ids: list[int],
    target_ids: list[int],
    target_pos: int,
    proposal_ids: list[int],
) -> list[float]:
    """Label proposal tokens by exact decoded-text containment in the target suffix.

    This keeps source-token verification token-level, but makes boundary
    supervision robust when a source token crosses the gold span boundary
    (for example, source token ".\n\n" while the target suffix only needs ".").
    """
    target_remaining = _decode_ids(tokenizer, target_ids[target_pos:])
    copied = ""
    labels: list[float] = []
    token_text_cache: dict[int, str] = {}

    for tok_id in proposal_ids:
        tok_id = int(tok_id)
        tok_text = token_text_cache.get(tok_id)
        if tok_text is None:
            tok_text = _decode_ids(tokenizer, [tok_id])
            token_text_cache[tok_id] = tok_text

        next_text = copied + tok_text
        if target_remaining.startswith(next_text):
            labels.append(1.0)
            copied = next_text
            continue

        # The source token would leave the gold text prefix. This covers both
        # ordinary first-mismatch negatives and boundary-crossing negatives.
        labels.append(0.0)
        break

    return labels


def train_bias_on_anchors(
    model: CopyConstraintModel,
    prefix_ids: torch.Tensor,
    source_ids: list[int],
    target_ids: list[int],
    anchors: list[tuple[int, int, int]],
    device: torch.device,
    bias_max_overcopy_extra: int,
    bias_pos_margin: float,
    bias_neg_margin: float,
    bias_neg_weight: float,
    bias_weight_tau: float,
    bias_text_boundary_labels: bool,
) -> torch.Tensor:
    if not anchors:
        return torch.zeros((), dtype=torch.float32, device=device)

    eps = torch.tensor(1e-8, dtype=torch.float32, device=device)
    pos_num = torch.zeros((), dtype=torch.float32, device=device)
    pos_den = torch.zeros((), dtype=torch.float32, device=device)
    neg_num = torch.zeros((), dtype=torch.float32, device=device)
    neg_den = torch.zeros((), dtype=torch.float32, device=device)
    weight_tau = max(float(bias_weight_tau), 1e-6)

    for target_pos, source_start, gold_len in anchors:
        max_extra = min(max(0, bias_max_overcopy_extra), len(source_ids) - source_start - gold_len)
        if max_extra > 0:
            max_extra = 1
        proposal_len = gold_len + max_extra
        proposal_ids = source_ids[source_start : source_start + proposal_len]
        if not proposal_ids:
            continue

        prefix_hidden, prefix_past = collect_proposal_prefix_state(
            model,
            prefix_ids,
            target_ids,
            target_pos,
            device,
        )
        predictor_states, logits = model.proposal_predictor_states_and_logits(
            prefix_hidden,
            prefix_past,
            proposal_ids,
            device=device,
        )
        raw_margins = proposal_source_margins(logits, proposal_ids, device=device)
        bias = model.source_token_bias(predictor_states, raw_margins)

        if bias_text_boundary_labels:
            label_values = build_text_boundary_accept_labels(
                model.tokenizer,
                source_ids,
                target_ids,
                target_pos,
                proposal_ids,
            )
            if not label_values:
                continue
            active_len = min(len(label_values), len(proposal_ids))
            accept_labels = torch.tensor(label_values[:active_len], dtype=torch.float32, device=device)
            predictor_states = predictor_states[:active_len]
            raw_margins = raw_margins[:active_len]
            bias = bias[:active_len]
        else:
            accept_labels = torch.zeros(len(proposal_ids), dtype=torch.float32, device=device)
            accept_labels[:gold_len] = 1.0
        adjusted_margins = raw_margins.float() + bias.float()

        pos_mask = accept_labels > 0.5
        neg_mask = ~pos_mask
        signed_raw = torch.where(pos_mask, raw_margins.float(), -raw_margins.float())
        hard_weights = torch.sigmoid(-signed_raw / weight_tau)

        if pos_mask.any():
            pos_adjusted = adjusted_margins[pos_mask]
            pos_weights = hard_weights[pos_mask]
            pos_hinge = torch.clamp(bias_pos_margin - pos_adjusted, min=0.0)
            pos_num = pos_num + (pos_weights * pos_hinge).sum()
            pos_den = pos_den + pos_weights.sum()
        if neg_mask.any():
            neg_adjusted = adjusted_margins[neg_mask]
            neg_weights = hard_weights[neg_mask]
            neg_hinge = torch.clamp(neg_adjusted + bias_neg_margin, min=0.0)
            neg_num = neg_num + (neg_weights * neg_hinge).sum()
            neg_den = neg_den + neg_weights.sum()

    losses: list[torch.Tensor] = []
    if pos_den.item() > 0:
        losses.append(pos_num / (pos_den + eps))
    if neg_den.item() > 0:
        losses.append(bias_neg_weight * (neg_num / (neg_den + eps)))

    if not losses:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(losses).sum()


def train_one_example(
    model: CopyConstraintModel,
    row: dict,
    device: torch.device,
    length_loss_weight: float,
    length_over_weight: float,
    length_loss_beta: float,
    min_match_length: int,
    bias_loss_weight: float,
    bias_pos_margin: float,
    bias_neg_margin: float,
    bias_neg_weight: float,
    bias_weight_tau: float,
    bias_max_overcopy_extra: int,
    bias_max_spans_per_example: int,
    bias_max_starts_per_anchor: int,
    bias_anchor_stride_divisor: int,
    bias_text_boundary_labels: bool,
    disable_thinking: bool,
) -> tuple[torch.Tensor, float, float, float, float]:
    tokenizer = model.tokenizer
    source_ids, target_ids = get_source_and_target_ids(tokenizer, row)
    if not target_ids:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, 0.0, 0.0, 0.0, 0.0

    instruction = row["instruction"]
    prefix_ids = encode_editor_prefix(
        tokenizer,
        row["context"],
        instruction,
        device,
        disable_thinking=disable_thinking,
    )
    state_tensor = collect_target_prefix_states(model, prefix_ids, target_ids, device)

    signals = row.get("__training_signals")
    if signals is None:
        signals = build_training_signals(source_ids, target_ids)
        row["__training_signals"] = signals
    match_t = torch.tensor(signals.match_lengths, dtype=torch.float32, device=device)
    longest_t = torch.tensor(signals.longest_copy_lengths, dtype=torch.float32, device=device)
    has_match_evidence = match_t >= float(min_match_length)
    trigger_targets = ((longest_t > 0) & has_match_evidence).long()

    trig_logits = model.trigger_logits(state_tensor, match_t).float()
    trig_loss = trigger_nll_loss(trig_logits, trigger_targets)

    positive_mask = (longest_t > 0) & has_match_evidence
    if positive_mask.any():
        length_pred = model.length_log(state_tensor[positive_mask], match_t[positive_mask]).float()
        len_loss = asymmetric_length_regression_loss(
            length_pred,
            longest_t[positive_mask],
            over_weight=length_over_weight,
            beta=length_loss_beta,
        )
        loss = trig_loss + (length_loss_weight * len_loss)
    else:
        len_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = trig_loss

    bias_loss = torch.zeros((), dtype=torch.float32, device=device)
    if bias_loss_weight > 0 and bias_max_spans_per_example != 0:
        anchors = build_bias_training_anchors(
            signals,
            source_ids,
            target_ids,
            min_match_length=min_match_length,
            max_spans=bias_max_spans_per_example,
            max_starts_per_anchor=bias_max_starts_per_anchor,
            stride_divisor=bias_anchor_stride_divisor,
        )
        if anchors:
            bias_loss = train_bias_on_anchors(
                model,
                prefix_ids,
                source_ids,
                target_ids,
                anchors,
                device=device,
                bias_max_overcopy_extra=bias_max_overcopy_extra,
                bias_pos_margin=bias_pos_margin,
                bias_neg_margin=bias_neg_margin,
                bias_neg_weight=bias_neg_weight,
                bias_weight_tau=bias_weight_tau,
                bias_text_boundary_labels=bias_text_boundary_labels,
            )
            loss = loss + (bias_loss_weight * bias_loss)

    with torch.no_grad():
        trig_acc = (trig_logits.argmax(dim=-1) == trigger_targets).float().mean().item()
    return loss, float(trig_loss.item()), float(len_loss.item()), float(bias_loss.item()), trig_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a constrained copy plugin with trigger + length heads.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="", help="Optional LoRA adapter path to load on top of --model-path.")
    parser.add_argument(
        "--train-jsonl",
        required=True,
        help="JSONL with context/instruction and target text. Tokenization is recomputed with the current tokenizer.",
    )
    parser.add_argument("--save-dir", default="outputs/checkpoints/copy_constraint_plugin")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--lr-scheduler-type", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--length-over-weight",
        type=float,
        default=1.0,
        help="Retained for CLI compatibility. The current length loss is symmetric, so 1.0 is the intended setting.",
    )
    parser.add_argument("--length-loss-beta", type=float, default=1.0)
    parser.add_argument("--min-match-length", type=int, default=5)
    parser.add_argument("--source-bias-scale", type=float, default=6.0)
    parser.add_argument("--bias-loss-weight", type=float, default=1.0)
    parser.add_argument("--bias-pos-margin", type=float, default=1.0)
    parser.add_argument("--bias-neg-margin", type=float, default=1.0)
    parser.add_argument("--bias-neg-weight", type=float, default=1.0)
    parser.add_argument("--bias-weight-tau", type=float, default=1.0)
    parser.add_argument("--bias-max-overcopy-extra", type=int, default=1)
    parser.add_argument("--bias-max-spans-per-example", type=int, default=50)
    parser.add_argument(
        "--bias-max-starts-per-anchor",
        type=int,
        default=3,
        help="Maximum number of equally-best source starts to supervise per bias anchor.",
    )
    parser.add_argument(
        "--bias-anchor-stride-divisor",
        type=int,
        default=1,
        help="Advance bias anchors by max(1, gold_len // divisor) to densify boundary coverage.",
    )
    parser.add_argument(
        "--bias-text-boundary-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decoded text-prefix labels for bias verifier supervision, adding boundary-crossing negatives.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '1'. Applied before torch import.",
    )
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.grad_accum_steps < 1:
        raise SystemExit("--grad-accum-steps must be >= 1")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    adapter_path = args.adapter_path or None

    model = CopyConstraintModel(
        args.model_path,
        adapter_path=adapter_path,
        freeze_base=args.freeze_base,
        torch_dtype=dtype,
        dropout=args.dropout,
        source_bias_scale=args.source_bias_scale,
    ).to(device)

    print(
        f"training copy-constrained plugin with model={args.model_path} "
        f"adapter={adapter_path or 'none'} "
        f"dtype={dtype} disable_thinking={args.disable_thinking} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise SystemExit("No trainable parameters found.")
    optimizer = AdamW(params, lr=args.lr)

    rows = load_rows(Path(args.train_jsonl), limit=args.max_train)
    if not rows:
        raise SystemExit("No training rows found.")

    updates_per_epoch = math.ceil(len(rows) / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, total_updates)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_rows = list(rows)
        if args.shuffle:
            random.Random(args.seed + epoch).shuffle(epoch_rows)

        running_loss = 0.0
        running_trigger_loss = 0.0
        running_length_loss = 0.0
        running_bias_loss = 0.0
        running_trigger_acc = 0.0
        updates_done = 0

        pbar = tqdm(epoch_rows, desc=f"epoch {epoch + 1}/{args.epochs}", unit="example")
        for step, row in enumerate(pbar, start=1):
            try:
                loss, trig_loss, len_loss, bias_loss, trig_acc = train_one_example(
                    model,
                    row,
                    device=device,
                    length_loss_weight=args.length_loss_weight,
                    length_over_weight=args.length_over_weight,
                    length_loss_beta=args.length_loss_beta,
                    min_match_length=args.min_match_length,
                    bias_loss_weight=args.bias_loss_weight,
                    bias_pos_margin=args.bias_pos_margin,
                    bias_neg_margin=args.bias_neg_margin,
                    bias_neg_weight=args.bias_neg_weight,
                    bias_weight_tau=args.bias_weight_tau,
                    bias_max_overcopy_extra=args.bias_max_overcopy_extra,
                    bias_max_spans_per_example=args.bias_max_spans_per_example,
                    bias_max_starts_per_anchor=args.bias_max_starts_per_anchor,
                    bias_anchor_stride_divisor=args.bias_anchor_stride_divisor,
                    bias_text_boundary_labels=args.bias_text_boundary_labels,
                    disable_thinking=args.disable_thinking,
                )
            except torch.OutOfMemoryError as exc:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise torch.OutOfMemoryError(
                    "CUDA OOM while processing "
                    f"{format_row_debug(row)} at epoch={epoch + 1} step={step}."
                ) from exc

            raw_loss = float(loss.item())
            (loss / args.grad_accum_steps).backward()

            should_step = (step % args.grad_accum_steps == 0) or (step == len(epoch_rows))
            if should_step:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates_done += 1

            running_loss += raw_loss
            running_trigger_loss += trig_loss
            running_length_loss += len_loss
            running_bias_loss += bias_loss
            running_trigger_acc += trig_acc
            pbar.set_postfix(
                loss=f"{running_loss / step:.4f}",
                trig=f"{running_trigger_loss / step:.4f}",
                len_loss=f"{running_length_loss / step:.4f}",
                bias=f"{running_bias_loss / step:.4f}",
                trig_acc=f"{running_trigger_acc / step:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                updates=updates_done,
            )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "plugin_state_dict": model.plugin_state_dict(),
            "model_path": args.model_path,
            "adapter_path": adapter_path,
            "dropout": args.dropout,
            "freeze_base": args.freeze_base,
            "length_loss_weight": args.length_loss_weight,
            "length_over_weight": args.length_over_weight,
            "length_loss_beta": args.length_loss_beta,
            "min_match_length": args.min_match_length,
            "source_bias_scale": args.source_bias_scale,
            "bias_loss_weight": args.bias_loss_weight,
            "bias_pos_margin": args.bias_pos_margin,
            "bias_neg_margin": args.bias_neg_margin,
            "bias_neg_weight": args.bias_neg_weight,
            "bias_weight_tau": args.bias_weight_tau,
            "bias_max_overcopy_extra": args.bias_max_overcopy_extra,
            "bias_max_spans_per_example": args.bias_max_spans_per_example,
            "bias_max_starts_per_anchor": args.bias_max_starts_per_anchor,
            "bias_anchor_stride_divisor": args.bias_anchor_stride_divisor,
            "bias_text_boundary_labels": args.bias_text_boundary_labels,
            "disable_thinking": args.disable_thinking,
        },
        save_dir / "plugin.pt",
    )
    print(f"saved -> {save_dir / 'plugin.pt'}")


if __name__ == "__main__":
    main()
