# Rebuttal Experiments

Run all commands from the repository root. `MODEL` is a base model directory
(Qwen3-4B, Qwen3-8B, or Ministral-8B-Instruct-2410), `PLUGIN` is a CopyGen
plugin trained for that model and domain, and `DOMAIN` is one of `contract`,
`literature`, `code`, `random_sequence`.

Predictions are scored with the standard scorer:

```bash
python scripts/baseline/evaluate_copy_outputs.py \
  --input PRED.jsonl \
  --oracle data/DOMAIN/test.jsonl \
  --summary-out SCORE.json
```

| Directory | Contents |
|---|---|
| `baseline/` | LLMA, CopySpec, PICARD, Longest-Continuation, span pointer, closed-model APIs |
| `robustness/` | Prompt variants and copy-exit probes |
| `engine/` | vLLM and SGLang integration, MLP-head overhead |

## Baselines

```bash
python scripts/rebuttal/baseline/eval_llma.py --model-path MODEL --oracle data/DOMAIN/test.jsonl \
  --output OUT/llma.jsonl --min-match-length 1 --max-match-length 5 --block-size 20
python scripts/rebuttal/baseline/eval_copyspec.py --model-path MODEL --oracle data/DOMAIN/test.jsonl \
  --output OUT/copyspec.jsonl --gamma 3 --copy-length 10
python scripts/rebuttal/baseline/eval_picard.py --model-path MODEL --oracle data/DOMAIN/test.jsonl \
  --output OUT/picard.jsonl --trigger-length 5 --copy-budget 16 --cooldown-tokens 1
python scripts/rebuttal/baseline/eval_longest_copy.py --model-path MODEL --oracle data/DOMAIN/test.jsonl \
  --output OUT/longest_copy.jsonl --trigger-length 5
```

For `random_sequence`, PICARD is run with `--reference-block-acceleration`.
Each run also writes `OUT/<name>.stats.json` with its throughput.

Span pointer (use 8192 instead of 4096 for `random_sequence`; throughput is
measured with `--max-samples 20`):

```bash
python scripts/rebuttal/baseline/train_span_pointer.py --model-path MODEL \
  --train-jsonl data/DOMAIN/train.jsonl --save-dir OUT/span_pointer
python scripts/rebuttal/baseline/eval_span_pointer.py --checkpoint OUT/span_pointer/span_pointer.pt \
  --oracle data/DOMAIN/test.jsonl --output OUT/span_pointer.jsonl \
  --max-actions 4096 --max-output-tokens 4096
```

Closed models, through an OpenAI-compatible endpoint (all four domains):

```bash
python scripts/rebuttal/baseline/eval_api.py --model gpt-4.1-nano-2025-04-14 --base-url API_BASE_URL \
  --api-key-env API_KEY --no-disable-thinking --output-dir OUT/api/gpt_4_1_nano
python scripts/rebuttal/baseline/eval_api.py --model gemini-2.5-flash-lite --base-url API_BASE_URL \
  --api-key-env API_KEY --output-dir OUT/api/gemini_2_5_flash_lite
python scripts/rebuttal/baseline/eval_api.py --model deepseek-v4-pro --base-url https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY --concurrency 32 --output-dir OUT/api/deepseek_v4_pro
```

## Decoding Controls

CopyGen:

```bash
python model/eval_copy_constrained.py --model-path MODEL --plugin PLUGIN \
  --input-jsonl data/DOMAIN/test.jsonl --output OUT/pred.jsonl --disable-thinking [OPTIONS]
```

Frozen base model:

```bash
python scripts/baseline/eval_llm_baseline.py --model-path MODEL --oracle data/DOMAIN/test.jsonl \
  --output OUT/base.jsonl --device cuda --stepwise --max-new-tokens 4096 --no-skip-special-tokens [OPTIONS]
```

| Experiment | OPTIONS |
|---|---|
| Fixed margin, raw x2 | `--margin-adjustment scale --heuristic-margin-scale 2 --heuristic-match-threshold 5` |
| Fixed margin, raw +1 | `--margin-adjustment shift --heuristic-margin-shift 1 --heuristic-match-threshold 5` |
| Fixed copy length 1 | `--fixed-copy-length 1` |
| Top-p sampling | `--temperature 0.7 --top-p 0.9 --seed SEED` (SEED = 42, 43, 44) |
| Min-p sampling | `--temperature 0.7 --min-p 0.05 --seed SEED` (SEED = 42, 43, 44) |

The margin and copy-length options apply to CopyGen only; sampling applies to both.

## Prompt Robustness

```bash
python -m scripts.rebuttal.robustness.build_prompt_variants --output-dir OUT/prompt_variants
```

This writes `DOMAIN.{para_a,para_b,tagged}.jsonl` for Contract, Literature, and
Code. Run CopyGen and the base model on each file with the commands above
(`--input-jsonl` / `--oracle`), and score against `data/DOMAIN/test.jsonl`,
adding `--output-tag COPYGEN_OUTPUT` for the tagged variant.

## Copy-Exit Probes

```bash
python -m scripts.rebuttal.robustness.exit_probe boundary --model-path MODEL \
  --plugin contract=PLUGIN_CONTRACT --plugin literature=PLUGIN_LITERATURE --plugin code=PLUGIN_CODE \
  --output-dir OUT/exit_probe/boundary
python -m scripts.rebuttal.robustness.exit_probe code-exit --model-path MODEL --plugin PLUGIN_CODE \
  --output-dir OUT/exit_probe/code_exit
python -m scripts.rebuttal.robustness.exit_probe continuation --model-path MODEL --plugin PLUGIN_CODE \
  --output-dir OUT/exit_probe/continuation
```

## Inference Engines

vLLM 0.25.1 environment:

```bash
VLLM_ENV=/path/to/vllm/env
SITE=$("$VLLM_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE/nvidia/cu13/lib:$SITE/torch/lib:$LD_LIBRARY_PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
PY=$VLLM_ENV/bin/python
```

SGLang 0.5.15.post1 environment (FlashInfer JIT needs a CUDA 12.9 `nvcc`):

```bash
SGLANG_ENV=/path/to/sglang/env
CUDA_TOOLKIT=/path/to/cuda-12.9
export PATH="$SGLANG_ENV/bin:$CUDA_TOOLKIT/bin:$PATH" CUDA_HOME="$CUDA_TOOLKIT"
export FLASHINFER_NVCC="$CUDA_TOOLKIT/bin/nvcc" PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
PY=$SGLANG_ENV/bin/python
```

If the default vLLM attention backend is unavailable on your GPU, add
`--attention-backend TRITON_ATTN` to the vLLM commands.

vLLM, full test sets (score `base.jsonl` / `copygen.jsonl` in each output directory):

```bash
$PY -m scripts.rebuttal.engine.vllm_batch --model-path MODEL --data data/DOMAIN/test.jsonl --domain DOMAIN \
  --plugin PLUGIN --arms base --max-num-seqs 1 --output-dir OUT/vllm_base/DOMAIN       # contract/literature/code
$PY -m scripts.rebuttal.engine.vllm_full_decode --model-path MODEL \
  --oracle DOMAIN=data/DOMAIN/test.jsonl --plugin DOMAIN=PLUGIN --output-dir OUT/vllm_copygen   # contract/literature/code
$PY -m scripts.rebuttal.engine.vllm_batch --model-path MODEL --data data/random_sequence/test.jsonl \
  --domain random_sequence --arms base --max-num-batched-tokens 16384 --output-dir OUT/vllm_stock/random_sequence
$PY -m scripts.rebuttal.engine.vllm_batch --model-path MODEL --data data/random_sequence/test.jsonl \
  --domain random_sequence --plugin PLUGIN --arms base copygen --max-num-batched-tokens 16384 \
  --output-dir OUT/vllm_batch8/random_sequence
```

`vllm_full_decode` accepts several `--oracle` / `--plugin` pairs in one run.

vLLM, batch-8 throughput on the first 24 examples (`stats.json` has tokens/s per arm):

```bash
$PY -m scripts.rebuttal.engine.vllm_batch --model-path MODEL --data data/DOMAIN/test.jsonl --domain DOMAIN \
  --plugin PLUGIN --arms base copygen --limit 24 --max-num-batched-tokens 32768 --max-model-len 32768 \
  --spec-tokens 64 --warmup-tokens 8 --gpu-memory-utilization 0.85 --output-dir OUT/vllm_batch8/DOMAIN
```

SGLang, full test sets (paired base and CopyGen):

```bash
$PY -m scripts.rebuttal.engine.sglang_client --model-path MODEL --data data/DOMAIN/test.jsonl \
  --plugin PLUGIN --output-dir OUT/sglang_client/DOMAIN
```

SGLang, batch-8 throughput (stock without `--plugin`; `--draft-tokens 128` for `random_sequence`):

```bash
$PY -m scripts.rebuttal.engine.sglang_native --model-path MODEL --data data/DOMAIN/test.jsonl \
  --output-dir OUT/sglang_native/DOMAIN_stock
$PY -m scripts.rebuttal.engine.sglang_native --model-path MODEL --data data/DOMAIN/test.jsonl \
  --plugin PLUGIN --draft-tokens 64 --output-dir OUT/sglang_native/DOMAIN_copygen
```

MLP-head overhead (run each command in its engine environment):

```bash
HEADS=(); for d in contract literature code; do HEADS+=(--oracle $d=data/$d/test.jsonl --plugin $d=PLUGIN_DIR/$d/plugin.pt); done
$PY -m scripts.rebuttal.engine.head_overhead --engine vllm --model-path MODEL "${HEADS[@]}" \
  --output OUT/head_overhead_vllm.json
$PY -m scripts.rebuttal.engine.head_overhead --engine sglang --model-path MODEL "${HEADS[@]}" \
  --output OUT/head_overhead_sglang.json
```
