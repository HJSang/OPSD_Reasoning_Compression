# CRISP: Compressed Reasoning via Iterative Self-Policy Distillation (Original OPSDC On-Policy Self-Distillation for Reasoning Compression)

This repository contains the code for **CRISP** (**C**ompressed **R**easoning via **I**terative **S**elf-**P**olicy Distillation), a method that teaches reasoning models to think more concisely by distilling their own concise behavior back into themselves.

**Paper:** [CRISP: Compressed Reasoning via Iterative Self-Policy Distillation](crisp_compressed_reasoning_via_iterative_self_policy_distillation.pdf) | [arXiv](https://arxiv.org/abs/2603.05433)

**Authors:** Hejian Sang\*, Yuanda Xu\*, Zhengze Zhou\*, Ran He\*, Zhipeng Wang, Jiachen Sun

**Related write-up:** [Scorer Choice in Math Reasoning Evaluation](https://zhengzezhou.github.io/math-scorer-choice/) — a four-policy decomposition of how verifier choice (answer-extraction vs. symbolic equivalence) can swing reported MATH-500 accuracy by up to ~80 percentage points on identical generations.

## Key Idea

Reasoning models think out loud, but much of what they say is noise. CRISP uses a single, almost trivial idea: *ask the model to be concise, then teach it to do so without being asked*.

- **Teacher**: The same model conditioned on a conciseness instruction (e.g., "Solve concisely, avoid unnecessary steps")
- **Student**: The same model without the conciseness instruction

Training generates student rollouts and minimizes per-token reverse KL divergence between student and teacher distributions. No ground-truth answers, no token budgets, no difficulty estimators.

## Results

All numbers are mean@8 under a 30K-token budget, scored with a dual-path grader (`Answer:` line **or** `\boxed{}`, math-verified). We use two conciseness instructions: **v1** (uniform "be concise") compresses more aggressively, while **v2** (difficulty-aware) better preserves accuracy on hard problems. Reduction is relative to the base model's average response length.

### MATH-500 (in-domain, easiest)

| Model | Base acc | v2 acc / reduction | v1 acc / reduction |
|-------|:--------:|:------------------:|:------------------:|
| Qwen3-8B  | 95.7 | 95.7 / 32% | 95.7 / 57% |
| Qwen3-14B | 93.0 | 95.2 / 35% | 96.3 / 56% |
| Qwen3-32B | 95.6 | 96.0 / 30% | 94.3 / 51% |
| DeepSeek-R1-Distill-Llama-8B | 71.3 | 79.8 / 23% | 82.1 / 32% |

### AIME 2024 (in-domain, harder)

| Model | Base acc | v2 acc / reduction | v1 acc / reduction |
|-------|:--------:|:------------------:|:------------------:|
| Qwen3-8B  | 76.2 | 75.0 / 17% | 72.9 / 33% |
| Qwen3-14B | 75.0 | 75.0 / 20% | 73.8 / 38% |
| Qwen3-32B | 80.5 | 80.4 / 19% | 72.9 / 30% |
| DeepSeek-R1-Distill-Llama-8B | 33.3 | 42.1 / −3% | 39.2 / 6% |

**Takeaways:**
- **Compression with preserved accuracy.** CRISP shortens reasoning by up to ~57% on MATH-500 while holding accuracy, and improves it where the base model has room (Qwen3-14B +3.3 pts on MATH-500; DeepSeek-R1-Distill-Llama-8B +10.8 pts).
- **Difficulty-adaptive.** Reductions are largest on the easier MATH-500 and smaller on the harder AIME benchmarks — the KL objective compresses easy problems more, with no explicit difficulty estimation.
- **Instruction-agnostic.** The effect does not depend on a single hand-tuned prompt: v1 trades more compression for a small accuracy cost, v2 protects hard problems; both compress strongly while preserving accuracy.
- **General capabilities and entropy preserved.** Out-of-domain accuracy on GPQA-Diamond and MMLU stays within ~1 point of the base model, and the student's policy entropy is stable throughout training (no entropy collapse).

## Repository Structure

```
OnPolicySD-open/
├── verl/                          # VERL framework (forked, with minor fixes)
├── workspace/
│   ├── config/
│   │   └── prompts.json           # Prompt templates (student, teacher, length prune)
│   ├── data/
│   │   ├── DAPO-Math-17k-dedup/   # Training data (17k math problems)
│   │   ├── MATH-500/              # Validation benchmark
│   │   ├── aime24/                # AIME 2024 validation
│   │   └── aime25/                # AIME 2025 validation
│   ├── src/
│   │   ├── data/
│   │   │   ├── process_eval_data.py          # Process eval datasets (train/val splits)
│   │   │   ├── prepare_length_prune_data.py  # Generate length pruning prompts
│   │   │   └── prepare_self_distill_data.py  # Generate self-distill prompts (with teacher solutions)
│   │   └── self_distill_hybrid/
│   │       ├── main_opsd.py       # OPSD entry point
│   │       ├── opsd_trainer.py    # OPSD trainer (JSD/reverse-KL loss)
│   │       ├── opsd_worker.py     # OPSD FSDP worker
│   │       ├── sd_worker.py       # Base self-distill worker
│   │       ├── sd_dataset.py      # Dataset for paired teacher/student prompts
│   │       └── sd_verifier.py     # Math answer verification
│   ├── scripts/sft/
│   │   └── train_opsd.sh          # Main training launch script
│   └── execution-configs/         # Hyperparameter configs for Qwen3-8B and 14B
```

## Setup

### Prerequisites

- 8x H100/H200 GPUs (80GB)
- Python 3.10+
- CUDA 12.4+

### Installation

```bash
git clone https://github.com/HJSang/OPSD_Reasoning_Compression.git
cd OPSD_Reasoning_Compression

# Install VERL and dependencies
cd verl
pip install -e .
cd ..

# Install additional dependencies
pip install sglang pandas datasets hydra-core omegaconf
```

## Quick Start

The full pipeline has 3 stages:

### Stage 1: Process Evaluation Data

Process DAPO-Math-17k-dedup into train/val splits and prepare validation benchmarks (MATH-500, AIME 2024, AIME 2025).

```bash
cd workspace/src/data

python process_eval_data.py \
    --data_dir ../../data \
    --output_dir ../../data/processed
```

This produces:
- `data/processed/train.parquet` — DAPO training split (95%)
- `data/processed/val_dapo.parquet` — DAPO validation split (5%)
- `data/processed/val_math500.parquet`, `val_aime24.parquet`, `val_aime25.parquet` — Evaluation benchmarks

### Stage 2: Generate Length Pruning Prompts

Create paired teacher/student prompts for OPSD training. The teacher prompt adds a conciseness instruction; the student prompt is the original DAPO-Math prompt unchanged.

```bash
# Batch mode (recommended) — generates all 4 variants with shared 80/20 split:
python prepare_length_prune_data.py batch \
    --input-parquet ../../data/DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet \
    --output-root ../../data

# This creates:
#   data/length_prune_concise/     — "Solve concisely" teacher prompt
#   data/length_prune_20pct/       — "Use 20% fewer tokens" teacher prompt
#   data/length_prune_50pct/       — "Use 50% fewer tokens" teacher prompt
#   data/length_prune_80pct/       — "Use 80% fewer tokens" teacher prompt
#
# Each directory contains:
#   self_distill_prompts.parquet       — Training prompts
#   self_distill_prompts_val.parquet   — Validation prompts
```

### Stage 3: Train OPSD

Launch OPSD training using the VERL HybridEngine (sglang for generation + FSDP for training).

#### Qwen3-8B

```bash
MODEL_PATH=/path/to/Qwen3-8B \
SD_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts.parquet \
SD_VAL_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts_val.parquet \
OPSD_BETA=0.5 \
SD_TEMPERATURE=1.0 \
SD_TOP_P=1.0 \
SD_MAX_TOKENS=8192 \
SFT_MAX_LENGTH=10240 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=32 \
MICRO_BATCH_SIZE=2 \
LEARNING_RATE=1e-6 \
TP_SIZE=2 \
GPU_MEM_UTIL=0.75 \
ULYSSES_SP_SIZE=4 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=30000 \
VAL_MAX_TOKENS=30000 \
CHECK_STRUCTURE=false \
USE_LIGER=true \
OPSD_LOSS_TYPE=reverse_kl \
TEACHER_UPDATE_FREQ=50 \
EXPERIMENT_NAME=opsd_length_prune_concise \
bash workspace/scripts/sft/train_opsd.sh
```

#### Qwen3-14B

```bash
MODEL_PATH=/path/to/Qwen3-14B \
SD_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts.parquet \
SD_VAL_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts_val.parquet \
OPSD_BETA=0.5 \
SD_TEMPERATURE=1.0 \
SD_TOP_P=1.0 \
SD_MAX_TOKENS=8192 \
SFT_MAX_LENGTH=10240 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=32 \
MICRO_BATCH_SIZE=2 \
LEARNING_RATE=1e-6 \
TP_SIZE=2 \
GPU_MEM_UTIL=0.75 \
ULYSSES_SP_SIZE=4 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=30000 \
VAL_MAX_TOKENS=30000 \
CHECK_STRUCTURE=false \
USE_LIGER=true \
OPSD_LOSS_TYPE=reverse_kl \
TEACHER_UPDATE_FREQ=50 \
EXPERIMENT_NAME=opsd_length_prune_concise \
bash workspace/scripts/sft/train_opsd.sh
```

Pre-configured hyperparameter files for various ablations (teacher update frequency, compression strength) are available in `workspace/execution-configs/`.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPSD_LOSS_TYPE` | `reverse_kl` | Loss type: `reverse_kl` or `jsd` |
| `OPSD_BETA` | `0.5` | JSD interpolation weight (only used when `jsd`) |
| `TEACHER_UPDATE_FREQ` | `50` | Steps between teacher weight updates (0 = frozen teacher) |
| `SD_TEMPERATURE` | `1.0` | Student rollout temperature |
| `SD_MAX_TOKENS` | `8192` | Max tokens for student generation |
| `SFT_MAX_LENGTH` | `10240` | Max sequence length for training |
| `CHECK_STRUCTURE` | `false` | Whether to require `<think>` tags in responses |
| `USE_LIGER` | `true` | Memory-efficient loss via logsumexp |

## How It Works

1. **Generate**: sglang produces student responses from question-only prompts
2. **Score**: Teacher forward pass computes logits on student-generated tokens using the conciseness-augmented prompt
3. **Train**: Minimize per-token reverse KL between student and teacher distributions on ALL responses (no correctness filtering)
4. **Sync**: Updated weights are automatically synced back to sglang for the next generation step
5. **Refresh teacher**: Every `TEACHER_UPDATE_FREQ` steps, copy student weights to teacher for progressive compression

## Acknowledgments

Built on top of [VERL](https://github.com/volcengine/verl) (HybridEngine for combined generation and training).

## Citation

```bibtex
@article{sang2025crisp,
  title={CRISP: Compressed Reasoning via Iterative Self-Policy Distillation},
  author={Sang, Hejian and Xu, Yuanda and Zhou, Zhengze and He, Ran and Wang, Zhipeng and Sun, Jiachen},
  journal={arXiv preprint arXiv:2603.05433},
  year={2026}
}
```
