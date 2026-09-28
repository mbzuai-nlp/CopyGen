# Scripts

This directory contains the support code used to build the benchmark, run baselines, and run analysis utilities.

## Directory Guide

- `scripts/fetch/`: optional raw-data download and extraction utilities for source corpora.
- `scripts/generate/`: benchmark construction scripts for the released domains.
- `scripts/baseline/`: plain LLM and LoRA baselines, plus exact-copy evaluation.

The released benchmark files already live under `data/`. Scripts that create new intermediate files or predictions should write them under `outputs/`.

## Main Commands

Train a CopyGen plugin:

```bash
python model/train_copy_constrained.py \
  --model-path MODEL_PATH \
  --train-jsonl data/random_sequence/train.jsonl \
  --save-dir outputs/checkpoints/random_sequence_plugin \
  --cuda-visible-devices 0 \
  --disable-thinking
```

Evaluate with copy-constrained decoding:

```bash
python model/eval_copy_constrained.py \
  --model-path MODEL_PATH \
  --plugin outputs/checkpoints/random_sequence_plugin/plugin.pt \
  --input-jsonl data/random_sequence/test.jsonl \
  --output outputs/predictions/random_sequence_predictions.jsonl \
  --cuda-visible-devices 0 \
  --disable-thinking
```

Run a plain LLM baseline:

```bash
python scripts/baseline/eval_llm_baseline.py \
  --model-path MODEL_PATH \
  --oracle data/random_sequence/test.jsonl \
  --output outputs/predictions/random_sequence_llm_baseline.jsonl \
  --max-new-tokens 8192
```

Score exact-copy outputs:

```bash
python scripts/baseline/evaluate_copy_outputs.py \
  --input outputs/predictions/random_sequence_predictions.jsonl \
  --oracle data/random_sequence/test.jsonl \
  --summary-out outputs/predictions/random_sequence_predictions.summary.json
```

Scripts for the experiments added during the rebuttal are described in `scripts/rebuttal/README.md`.
