# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CRISP / OPSD (On-Policy Self-Distillation) trains reasoning LLMs to think more concisely by distilling their own concise behavior back into themselves. There is **one teacher and one student that are the same base model**, differing only by prompt:

- **Student** generates a rollout from the *question-only* prompt (`sft_prompt`).
- **Teacher** (frozen ref model) re-scores those *same response tokens* under a longer *conciseness* prompt (`sd_prompt`, e.g. "Solve concisely").
- Training minimizes per-token **reverse-KL** (default) or **JSD** between student and teacher logits on the response positions, over **ALL** rollouts (no correctness filtering — verification is metrics-only).

No ground-truth answers, token budgets, or difficulty estimators are used in the loss.

## Commands

All paths below are relative to `workspace/`.

```bash
# --- Environment (assumes a cluster image with verl/torch/sglang preinstalled) ---
bash scripts/sft/setup_sft.sh          # adds the two extras (math-verify, tensordict) + prints versions
# Fresh install instead: pip install -r requirements.txt   (pins verl@deec5d02, torch 2.9.1, sglang 0.5.9)

# --- Unit tests (CPU, fast) ---
pytest src/self_distill_hybrid/test_opsd_jsd.py            # JSD/reverse-KL/entropy loss correctness + chunk invariance
pytest src/self_distill_hybrid/test_opsd_jsd.py::TestJSDLossEquivalence::test_loss_values_match   # single test

# --- Data pipeline (run in order) ---
cd src/data
python process_eval_data.py --data_dir ../../data --output_dir ../../data/processed   # DAPO train/val split + MATH-500/AIME parquets
python prepare_length_prune_data.py batch \                # builds 4 teacher-strength variants (concise/20/50/80pct)
    --input-parquet ../../data/DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet \
    --output-root ../../data
# `single` subcommand builds one variant: --teacher-style {concise,percent_reduce} --percent-reduce N

# --- Training (8x H100/H200 80GB). See README "Quick Start" for the full env-var block ---
MODEL_PATH=/path/to/Qwen3-8B \
SD_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts.parquet \
OPSD_LOSS_TYPE=reverse_kl TEACHER_UPDATE_FREQ=50 ... \
bash workspace/scripts/sft/train_opsd.sh

# --- Checkpoint export (FSDP shards -> HF format) ---
bash scripts/sft/merge_checkpoints.sh
```

`train_opsd.sh` is the single entry point: it runs `process_eval_data.py` on-cluster, auto-detects val parquets, then launches `python -m self_distill_hybrid.main_opsd` with a long list of Hydra overrides. Tune behavior via the env vars it reads (documented in the script header and the README "Key Hyperparameters" table), not by editing the python.

## Architecture

Built on a **pinned fork of [VERL](https://github.com/volcengine/verl)** (`deec5d02`, between v0.7.0 and v0.7.1) using its **HybridEngine**: sglang for generation and FSDP for training are *colocated* on the same GPUs, with weights synced between them each step. The repo adds OPSD-specific subclasses rather than patching verl in-tree; the custom math scorer is loaded via verl's `custom_reward_function.path`, not an overlay.

The training stack (`workspace/src/self_distill_hybrid/`) layers on verl:

- **`main_opsd.py`** — Hydra/Ray entry point. Maps the worker to `Role.ActorRolloutRef` (not `ActorRollout`) **specifically so the ref model is materialized** — that ref model *is* the frozen teacher. Builds the dataset, an optional generation-based val dataset + reward manager, then runs the trainer.
- **`opsd_trainer.py` (`OPSDTrainer`)** — the per-step loop in `fit()`:
  1. **Generate**: swaps `raw_prompt` (initially the teacher `sd_prompt`) to `sft_prompt` (question-only) and generates student rollouts via sglang; then `sleep_replicas()` to free rollout GPU memory for the backward pass.
  2. **Verify**: `verify_batch` scores correctness for metrics only — results never filter the batch.
  3. **Train**: `build_opsd_batch` + dispatch to `update_opsd`.
  4. **Sync**: `CheckpointEngineManager.update_weights()` pushes fresh student weights into sglang so the next generation is on-policy. Skipping this would break the on-policy assumption.
- **`opsd_worker.py` (`OPSDWorker`)** — subclasses verl's `AsyncActorRolloutRefWorker`. Adds two registered methods:
  - `update_opsd`: two forward passes per micro-batch — teacher (`ref_module_fsdp`, no-grad) and student (`actor_module_fsdp`, with-grad) — then the divergence loss on response logits. Has padded and unpadded (flash-attn varlen) logit paths, and `_liger` loss variants (logsumexp mixture + progressive teacher-chunk freeing) for lower peak memory.
  - `update_teacher`: optional hard-copy of student shards into the ref model every `TEACHER_UPDATE_FREQ` steps (progressive compression). Copies FSDP shards **in lock-step without `summon_full_params`** — full materialization OOMs Qwen3-14B on 80GB.
- **`sd_dataset.py` / `sd_verifier.py`** — dataset loading + the batch builders and the dual-path math verifier.

### The load-bearing invariant

Teacher and student sequences share the **same response tokens** but have **different (different-length) prompts**. The reverse-KL/JSD loss aligns teacher vs. student response logits **by position**, so per-sample response-token counts must match exactly between the two sides. A one-sided truncation would silently misalign every subsequent sample in the flattened batch. This is enforced in two places, and **must stay enforced** when touching the data path:

1. `SelfDistillDataset` (`sd_dataset.py`) drops rows whose teacher prompt exceeds `data.max_prompt_length`.
2. `build_opsd_batch` / `_tokenize_sequence` (`sd_verifier.py`) **refuse** (return `None` → drop the pair) any sequence over `opsd.sft_max_length` rather than truncating.

The `assert teacher_logits.shape[0] == student_logits.shape[0]` in `opsd_worker._opsd_training_step` is a defensive cross-check, not the primary guarantee. If a large fraction of samples is being silently dropped, **raise `SFT_MAX_LENGTH`** — don't relax the drop.

### Data columns (parquet → batch)

`prepare_length_prune_data.py` emits per row:
- **`sft_prompt`** = student prompt = original DAPO-Math question, unchanged → used for **generation** and as the **student** logit prompt.
- **`sd_prompt`** = teacher prompt = question + conciseness instruction (from `config/prompts.json`) → used only as the **teacher** logit prompt.
- `ground_truth` (verification), `question` (logging), `teacher_solution` (empty for length-pruning).

Prompt templates live in `workspace/config/prompts.json` (`length_prune_teacher`, `length_prune_teacher_percent_reduce`, `opsd_qwen3_student`, etc.). Qwen3 thinking mode is pinned via `data.apply_chat_template_kwargs.enable_thinking: true` in the config.

### Verification / scoring

Both the in-trainer verifier (`sd_verifier.verify_response`) and the val reward fn (`src/rewards/dual_path_math_verify.py`) use **dual-path math_verify**: (1) regex-extract `Answer: X` → wrap in `\boxed{}` → sympy symbolic-equivalence; (2) fallback math_verify over the full response to catch in-prose `\boxed{...}` from Qwen3 thinking mode. Correct iff *either* path matches. The reward fn dispatches only `math*`/`aime*` data_sources to this scorer and delegates everything else to verl's `default_compute_score`.

## Gotchas

- **Do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`** — sglang's `torch_memory_saver` refuses to initialize under that allocator and crashes rollout init (see the comment block atop `train_opsd.sh`). Address reverse-KL/JSD OOMs by lowering the `chunk_size` in the loss functions in `opsd_worker.py` instead, or halving `MICRO_BATCH_SIZE`.
- **DP padding uses `dp_world = total_gpus // ulysses_sp`, not `total_gpus`** (TP is rollout-only and doesn't collapse the actor's DP axis). The OPSD batch is padded to a multiple of `dp_world` before dispatch; getting this wrong yields `AssertionError("only support equal chunk")`.
- `execution-configs/` holds the per-ablation hyperparameter sets (Qwen3-8B/14B × teacher-update-freq `tu1`..`tu100` × compression strength). These are the canonical reproductions — prefer them over hand-rolled env vars.
- `*.pyc` under `src/.../__pycache__/` for cpython-312 **and** 314 are checked in; ignore them.
