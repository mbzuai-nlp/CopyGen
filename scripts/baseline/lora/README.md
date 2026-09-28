# LoRA Baseline

This baseline trains a supervised LoRA adapter with the same prompt format used by CopyGen, then evaluates it with greedy decoding.

Train:

```bash
python scripts/baseline/lora/train_lora_copy_baseline.py \
  --model-path MODEL_PATH \
  --train-jsonl data/random_sequence/train.jsonl \
  --output-dir outputs/checkpoints/lora/random_sequence \
  --epochs 1 \
  --max-seq-length 8192
```

Evaluate:

```bash
python scripts/baseline/lora/eval_lora_copy_baseline.py \
  --model-path MODEL_PATH \
  --adapter-path outputs/checkpoints/lora/random_sequence \
  --oracle data/random_sequence/test.jsonl \
  --output outputs/predictions/random_sequence_lora_baseline.jsonl \
  --max-new-tokens 8192
```

Score:

```bash
python scripts/baseline/evaluate_copy_outputs.py \
  --input outputs/predictions/random_sequence_lora_baseline.jsonl \
  --oracle data/random_sequence/test.jsonl \
  --summary-out outputs/predictions/random_sequence_lora_baseline.summary.json
```
