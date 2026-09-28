#!/usr/bin/env python3
"""Ablation entry point for copy-constrained plugin training.

This wrapper keeps ``train_copy_constrained.py`` as the source of truth and only
changes the loss term being ablated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _extract_ablation(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ablation",
        choices=["no_adaptive_weighting"],
        required=True,
    )
    args, remaining = parser.parse_known_args(argv[1:])
    return args.ablation, [argv[0], *remaining]


def _install_no_adaptive_weighting_patch(base_module) -> None:
    import torch

    def train_bias_on_anchors(
        model,
        prefix_ids,
        source_ids,
        target_ids,
        anchors,
        device,
        bias_max_overcopy_extra,
        bias_pos_margin,
        bias_neg_margin,
        bias_neg_weight,
        bias_weight_tau,
        bias_text_boundary_labels,
    ):
        if not anchors:
            return torch.zeros((), dtype=torch.float32, device=device)

        eps = torch.tensor(1e-8, dtype=torch.float32, device=device)
        pos_num = torch.zeros((), dtype=torch.float32, device=device)
        pos_den = torch.zeros((), dtype=torch.float32, device=device)
        neg_num = torch.zeros((), dtype=torch.float32, device=device)
        neg_den = torch.zeros((), dtype=torch.float32, device=device)

        for target_pos, source_start, gold_len in anchors:
            max_extra = min(max(0, bias_max_overcopy_extra), len(source_ids) - source_start - gold_len)
            if max_extra > 0:
                max_extra = 1
            proposal_len = gold_len + max_extra
            proposal_ids = source_ids[source_start : source_start + proposal_len]
            if not proposal_ids:
                continue

            prefix_hidden, prefix_past = base_module.collect_proposal_prefix_state(
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
            raw_margins = base_module.proposal_source_margins(logits, proposal_ids, device=device)
            bias = model.source_token_bias(predictor_states, raw_margins)

            if bias_text_boundary_labels:
                label_values = base_module.build_text_boundary_accept_labels(
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
                raw_margins = raw_margins[:active_len]
                bias = bias[:active_len]
            else:
                accept_labels = torch.zeros(len(proposal_ids), dtype=torch.float32, device=device)
                accept_labels[:gold_len] = 1.0

            adjusted_margins = raw_margins.float() + bias.float()
            pos_mask = accept_labels > 0.5
            neg_mask = ~pos_mask

            # Ablation: remove adaptive hard/easy weighting and give every bias
            # verifier token the same training weight.
            uniform_weights = torch.ones_like(raw_margins.float())

            if pos_mask.any():
                pos_adjusted = adjusted_margins[pos_mask]
                pos_weights = uniform_weights[pos_mask]
                pos_hinge = torch.clamp(bias_pos_margin - pos_adjusted, min=0.0)
                pos_num = pos_num + (pos_weights * pos_hinge).sum()
                pos_den = pos_den + pos_weights.sum()
            if neg_mask.any():
                neg_adjusted = adjusted_margins[neg_mask]
                neg_weights = uniform_weights[neg_mask]
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

    base_module.train_bias_on_anchors = train_bias_on_anchors


def main() -> None:
    ablation, train_argv = _extract_ablation(sys.argv)
    sys.argv = train_argv

    import model.train_copy_constrained as base_train

    if ablation == "no_adaptive_weighting":
        _install_no_adaptive_weighting_patch(base_train)
    else:
        raise SystemExit(f"unknown ablation: {ablation}")

    base_train.main()


if __name__ == "__main__":
    main()
