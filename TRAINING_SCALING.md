# MetaKube Training Configuration and Hardware Scaling

This guide explains how to run LoRA/QLoRA training without hardcoding the codebase for one machine. Use it when moving from a small Kaggle/Colab smoke test to larger models or stronger GPUs.

## Core Rule

Keep the training code unchanged.

Change training behavior through:

1. `config/config.yaml` for shared/default settings.
2. CLI flags for one-off runs, notebooks, and hardware-specific experiments.

The main training script is:

```bash
python train/sft_lora.py
```

It reads defaults from:

```text
config/config.yaml
```

under:

```yaml
sft:
```

## Quick Start

Small smoke test on Kaggle/Colab T4:

```bash
python data/synthetic_kfrd.py

python train/sft_lora.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path data/datasets/kfrd/kfrd_sft.json \
  --output_dir data/models/kubellm \
  --lora_rank 8 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --num_epochs 1 \
  --precision fp16 \
  --load_in_4bit

python eval/run_lora_smoke.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --lora_path data/models/kubellm
```

This validates the train-save-load-generate loop before larger training.

## Main Settings

| Setting | Where | What It Controls | Change First When |
|---|---|---|---|
| `model_id` | config or CLI | Base model to train | Scaling model size |
| `dataset_path` | config or CLI | Training JSON file | Switching datasets |
| `output_dir` | config or CLI | Where adapter is saved | Running on Kaggle/Colab outputs |
| `precision` | config or CLI | `auto`, `fp16`, `bf16`, `fp32` | Changing GPU type |
| `load_in_4bit` | config or CLI | Enables QLoRA | Low VRAM or 7B+ models |
| `batch_size` | config or CLI | Per-device batch size | OOM or faster GPUs |
| `gradient_accumulation_steps` | config or CLI | Effective batch size without more VRAM | Batch size is low |
| `max_length` | config or CLI | Token context length | OOM or better GPUs |
| `lora_rank` | config or CLI | LoRA capacity | Scaling quality/capacity |
| `lora_alpha` | config or CLI | LoRA scaling | Usually `2 * lora_rank` |
| `gradient_checkpointing` | config or CLI | Saves memory, costs speed | OOM on larger models |
| `device_map` | config or CLI | Model placement | Multi-GPU or large models |

## Config vs CLI

Use `config/config.yaml` for values you want to keep across runs:

```yaml
sft:
  model_id: "Qwen/Qwen3-8B-Instruct"
  dataset_path: "data/datasets/kfrd/kfrd_sft.json"
  output_dir: "data/models/kubellm"
  precision: "auto"
  load_in_4bit: true
  batch_size: 1
  gradient_accumulation_steps: 8
  max_length: 1024
```

Use CLI overrides for hardware-specific or experimental changes:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --precision fp16 \
  --batch_size 1
```

Do not edit core code just to change model size, precision, batch size, sequence length, or LoRA rank.

## Recommended Hardware Profiles

### MacBook / Low-Memory Local Machine

Use this only for code checks, dataset generation, and tiny CPU/GPU smoke tests. Do not train 7B/8B models on an 8GB MacBook.

Recommended:

```bash
python data/synthetic_kfrd.py
python train/sft_lora.py --help
python eval/run_lora_smoke.py --help
```

If you must test training locally:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --num_epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 512 \
  --precision fp32 \
  --no-load_in_4bit
```

Expect this to be slow. Prefer Kaggle/Colab.

### Kaggle / Colab T4

Good for correctness validation and small-to-medium experiments.

Recommended first run:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --lora_rank 8 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --num_epochs 1 \
  --precision fp16 \
  --load_in_4bit
```

Next step:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --lora_rank 8 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --num_epochs 1 \
  --precision fp16 \
  --load_in_4bit
```

For 7B/8B on T4, use conservative settings:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --lora_rank 8 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --num_epochs 1 \
  --precision fp16 \
  --load_in_4bit \
  --gradient_checkpointing
```

Notes:

- T4 generally uses `fp16`, not `bf16`.
- Dual T4 does not automatically behave like one larger GPU.
- If training 7B/8B on T4, expect slow runs and possible OOM.

### Higher-VRAM GPUs

Examples: A100, H100, L40S, A6000, RTX 4090.

Use `precision auto` or `bf16` if supported:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen3-8B-Instruct \
  --lora_rank 32 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --num_epochs 2 \
  --precision auto \
  --load_in_4bit
```

If VRAM is comfortable, try regular LoRA instead of QLoRA:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen3-8B-Instruct \
  --lora_rank 64 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_length 2048 \
  --precision bf16 \
  --no-load_in_4bit
```

For very strong infrastructure, scale in this order:

1. Increase `max_length`.
2. Increase `batch_size`.
3. Increase `lora_rank`.
4. Increase `num_epochs`.
5. Disable `load_in_4bit` only if memory is sufficient.

## Safe Scaling Path

Use this progression:

| Stage | Model | Goal | Suggested Settings |
|---|---|---|---|
| 1 | `Qwen2.5-0.5B-Instruct` | Verify pipeline | rank 8, length 1024, 1 epoch |
| 2 | `Qwen2.5-1.5B-Instruct` | Verify quality improves | rank 8-16, length 1024 |
| 3 | `Qwen2.5-3B-Instruct` | Medium test | rank 16, length 1024-2048 |
| 4 | 7B/8B | Large QLoRA | rank 8-32, length 1024 first |
| 5 | 7B/8B on strong GPU | Higher quality | rank 32-64, length 2048-4096 |

Do not start with 7B/8B until the small model can:

- train successfully
- save an adapter
- load the adapter
- generate a Kubernetes diagnosis
- produce valid-looking `kubectl` commands

## What To Change First When OOM Happens

OOM usually means VRAM ran out. Fix it in this order:

1. Reduce `max_length`.
2. Keep `batch_size=1`.
3. Increase `gradient_accumulation_steps` instead of batch size.
4. Enable `load_in_4bit`.
5. Enable `gradient_checkpointing`.
6. Reduce `lora_rank`.
7. Use a smaller model.

Example OOM recovery:

```bash
python train/sft_lora.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 512 \
  --lora_rank 8 \
  --precision fp16 \
  --load_in_4bit \
  --gradient_checkpointing
```

## Common Problems

| Problem | Likely Cause | Fix |
|---|---|---|
| CUDA OOM at model load | Model too large or not quantized | Use `--load_in_4bit`, smaller model |
| CUDA OOM during training | Batch/length too high | Lower `max_length`, keep `batch_size=1` |
| T4 BF16 error | T4 does not support BF16 well | Use `--precision fp16` |
| Adapter trains but eval fails | Using Ollama eval on Kaggle | Use `eval/run_lora_smoke.py` |
| Bad `kubectl` commands | Dataset issue | Regenerate `data/synthetic_kfrd.py` output |
| Training very slow | Gradient checkpointing/large model | Accept for low VRAM, or use stronger GPU |
| No adapter files saved | Wrong `output_dir` | Save under a persistent Kaggle/Colab output path |

## Validation Commands

Before training:

```bash
python train/sft_lora.py --help
python eval/run_lora_smoke.py --help
```

After training:

```bash
python eval/run_lora_smoke.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --lora_path data/models/kubellm
```

Check that the response contains:

- a plausible root cause
- concrete `kubectl` commands
- no obvious namespace/pod corruption like `-n pod-abc123`

## Keeping The Pipeline Hardware-Agnostic

Follow these rules:

1. Do not put GPU-specific assumptions inside training code.
2. Keep defaults in `config/config.yaml`.
3. Use CLI overrides for notebook-specific runs.
4. Prefer `precision: auto` in config, and explicit `--precision fp16` only for T4 runs.
5. Keep `load_in_4bit` configurable.
6. Test small before scaling large.
7. Save every trained adapter and run the smoke test before changing model size.

The same script should work from small T4 validation to larger GPU training by changing only config values or CLI flags.
