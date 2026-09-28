#!/usr/bin/env python3
"""
python scripts/baseline/lora/train_lora_copy_baseline.py \
  --model-path MODEL_PATH \
  --train-jsonl data/random_sequence/train.jsonl \
  --output-dir outputs/checkpoints/lora/random_sequence \
  --epochs 1 \
  --max-seq-length 8192 \
  --gradient-accumulation-steps 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.runtime_utils import apply_cuda_visible_devices_from_argv

apply_cuda_visible_devices_from_argv(sys.argv, default="1")

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from model.prompt_utils import render_editor_prompt
except ModuleNotFoundError:
    from prompt_utils import render_editor_prompt


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


def encode_supervised_example(
    tokenizer,
    row: dict,
    max_seq_length: int,
) -> dict[str, list[int]] | None:
    prompt = render_editor_prompt(tokenizer, row["context"], row["instruction"])
    target = get_target_text(row)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if len(input_ids) > max_seq_length:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class SupervisedCopyDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_seq_length: int):
        self.examples: list[dict[str, list[int]]] = []
        skipped = 0
        for row in rows:
            encoded = encode_supervised_example(tokenizer, row, max_seq_length=max_seq_length)
            if encoded is None:
                skipped += 1
                continue
            self.examples.append(encoded)
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


@dataclass
class SupervisedDataCollator:
    tokenizer: object

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA baseline on copy-task gold answers.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True, help="Task/oracle JSONL with context/instruction/gold_answer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module name fragments for LoRA injection",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '1'. Applied before torch import.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.train_jsonl), limit=args.max_train)
    if not rows:
        raise SystemExit("No training rows found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SupervisedCopyDataset(rows, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    if len(dataset) == 0:
        raise SystemExit("All rows were skipped, likely due to --max-seq-length being too small.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "train_rows_requested": len(rows),
        "train_rows_used": len(dataset),
        "train_rows_skipped": dataset.skipped,
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [m.strip() for m in args.target_modules.split(",") if m.strip()],
    }
    (output_dir / "train_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved adapter -> {output_dir}")


if __name__ == "__main__":
    main()
