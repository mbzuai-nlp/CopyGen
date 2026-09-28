#!/usr/bin/env python3
"""
Train the span-pointer heads on oracle-segmented targets with the LM frozen.

Each target is split into GEN/COPY/STOP actions from difflib.SequenceMatcher blocks between
the tokenized context and answer (blocks of >= min_copy_tokens become COPY actions). The loss
is the gate cross-entropy at every action boundary plus the span NLL of every COPY action,
normalized over all context spans, averaged over actions.

Example:
  python scripts/rebuttal/baseline/train_span_pointer.py \
    --model-path MODEL_PATH \
    --train-jsonl data/contract/train.jsonl \
    --save-dir outputs/rebuttal/span_pointer/qwen3_8b/contract \
    --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv
from model.prompt_utils import encode_editor_prefix

apply_cuda_visible_devices_from_argv(sys.argv)

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler

from scripts.rebuttal.baseline.span_pointer import SpanPointerModel, all_span_logsumexp

COPY_CHUNK_SIZE = 16


@dataclass(frozen=True)
class Action:
    mode: str  # GEN, COPY or STOP
    target_start: int
    source_start: int = -1
    length: int = 0


def oracle_actions(source_ids: list[int], target_ids: list[int], min_copy_tokens: int) -> list[Action]:
    actions: list[Action] = []
    cursor = 0
    for block in SequenceMatcher(None, source_ids, target_ids, autojunk=False).get_matching_blocks():
        if block.size >= min_copy_tokens:
            actions.extend(Action("GEN", position) for position in range(cursor, block.b))
            actions.append(Action("COPY", block.b, block.a, block.size))
            cursor = block.b + block.size
    actions.extend(Action("GEN", position) for position in range(cursor, len(target_ids)))
    actions.append(Action("STOP", len(target_ids)))

    replay: list[int] = []
    for action in actions:
        if action.mode == "GEN":
            replay.append(target_ids[action.target_start])
        elif action.mode == "COPY":
            replay.extend(source_ids[action.source_start : action.source_start + action.length])
    if replay != target_ids:
        raise ValueError("oracle actions do not reproduce the target")
    return actions


def load_examples(path: Path, tokenizer, min_copy_tokens: int, limit: int = 0) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_ids = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(row.get("gold_answer", row.get("edit_content", "")), add_special_tokens=False)[
                "input_ids"
            ]
            actions = oracle_actions(source_ids, target_ids, min_copy_tokens)
            examples.append({"row": row, "source_ids": source_ids, "target_ids": target_ids, "actions": actions})
            if limit and len(examples) >= limit:
                break
    return examples


def example_loss(model: SpanPointerModel, example: dict, device: torch.device, disable_thinking: bool) -> torch.Tensor:
    row = example["row"]
    target_ids = example["target_ids"]
    actions = example["actions"]
    prefix_ids = encode_editor_prefix(
        model.tokenizer, row["context"], row["instruction"], device, disable_thinking=disable_thinking
    )
    source_states = model.encode_source(torch.tensor([example["source_ids"]], dtype=torch.long, device=device))

    # teacher-forced decoder state before every answer token and before EOS
    teacher_ids = torch.cat([prefix_ids, torch.tensor([target_ids], dtype=torch.long, device=device)], dim=1)
    hidden, _ = model.hidden_states(teacher_ids)
    prefix_length = prefix_ids.shape[1]
    decoder_states = hidden[0, prefix_length - 1 : prefix_length + len(target_ids)]

    boundaries = torch.tensor([action.target_start for action in actions], dtype=torch.long, device=device)
    boundary_states = decoder_states.index_select(0, boundaries)
    gate_labels = torch.tensor([int(action.mode == "COPY") for action in actions], dtype=torch.long, device=device)
    loss_sum = F.cross_entropy(model.gate_logits(boundary_states).float(), gate_labels, reduction="none").sum()

    copy_rows = [index for index, action in enumerate(actions) if action.mode == "COPY"]
    if copy_rows:
        copy_states = boundary_states.index_select(0, torch.tensor(copy_rows, dtype=torch.long, device=device))
        start_chunks, end_chunks = [], []
        for chunk_start in range(0, len(copy_rows), COPY_CHUNK_SIZE):
            start_scores, end_scores = model.endpoint_scores(
                copy_states[chunk_start : chunk_start + COPY_CHUNK_SIZE], source_states
            )
            start_chunks.append(start_scores)
            end_chunks.append(end_scores)
        start_scores = torch.cat(start_chunks, dim=0)
        end_scores = torch.cat(end_chunks, dim=0)
        copies = [actions[index] for index in copy_rows]
        starts = torch.tensor([action.source_start for action in copies], dtype=torch.long, device=device)
        ends = torch.tensor([action.source_start + action.length - 1 for action in copies], dtype=torch.long, device=device)
        rows = torch.arange(len(copies), device=device)
        gold_scores = start_scores[rows, starts].float() + end_scores[rows, ends].float()
        loss_sum = loss_sum + (all_span_logsumexp(start_scores, end_scores) - gold_scores).sum()
    return loss_sum / len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train oracle-segmented span-pointer heads on a frozen LM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-copy-tokens", type=int, default=2)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = SpanPointerModel(args.model_path)
    model.to(device)
    model.train()
    examples = load_examples(Path(args.train_jsonl), model.tokenizer, args.min_copy_tokens, limit=args.max_train)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_updates = math.ceil(len(examples) / args.grad_accum_steps) * args.epochs
    warmup_steps = math.ceil(total_updates * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    print(
        f"model={args.model_path} examples={len(examples)} updates={total_updates} warmup={warmup_steps} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    update = 0

    def optimizer_step(accumulated: int, loss_sum: float, log) -> None:
        nonlocal update
        if accumulated < args.grad_accum_steps:
            # last partial accumulation: rescale to a mean over the examples it contains
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(args.grad_accum_steps / accumulated)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        record = {
            "update": update,
            "loss": loss_sum / accumulated,
            "grad_norm": float(grad_norm),
            "lr": scheduler.get_last_lr()[0],
        }
        log.write(json.dumps(record) + "\n")
        log.flush()

    optimizer.zero_grad(set_to_none=True)
    with (save_dir / "training_metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            order = list(examples)
            random.Random(args.seed + epoch).shuffle(order)
            accumulated = 0
            loss_sum = 0.0
            progress = tqdm(order, desc=f"epoch {epoch + 1}/{args.epochs}", unit="example")
            for example in progress:
                loss = example_loss(model, example, device, args.disable_thinking)
                (loss / args.grad_accum_steps).backward()
                accumulated += 1
                loss_sum += float(loss.detach().item())
                progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}", updates=update)
                if accumulated == args.grad_accum_steps:
                    optimizer_step(accumulated, loss_sum, log)
                    accumulated = 0
                    loss_sum = 0.0
            if accumulated:
                optimizer_step(accumulated, loss_sum, log)

    config = {key: value for key, value in vars(args).items() if key != "cuda_visible_devices"}
    config.update(train_examples=len(examples), updates=update, warmup_steps=warmup_steps)
    model.save_heads(save_dir / "span_pointer.pt", config)
    print(f"saved span-pointer heads -> {save_dir / 'span_pointer.pt'}")


if __name__ == "__main__":
    main()
