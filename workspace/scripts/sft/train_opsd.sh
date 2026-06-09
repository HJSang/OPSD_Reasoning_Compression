#!/bin/bash
# =============================================================================
# OPSD (On-Policy Self-Distillation) Training Script
#
# Uses VERL's HybridEngine with JSD divergence between teacher and student.
# Teacher = frozen ref model (initial weights), Student = trainable actor model.
#
# Key differences from train_self_distill_hybrid.sh:
#   - Uses main_opsd module instead of main_self_distill
#   - Uses opsd_trainer config instead of sd_trainer
#   - Trains on ALL student rollouts (no correctness filtering)
#   - Loss is JSD between teacher and student logit distributions
#   - No BASELINE_SFT_MODE (not applicable to OPSD)
#   - Adds OPSD_BETA parameter for JSD interpolation
#
# Flow per training step:
#   1. Generate: sglang produces student responses from question-only prompt
#   2. Verify: check math correctness for metrics (but do NOT filter)
#   3. Train: JSD update on ALL responses using frozen teacher ref model
#   -> Weights automatically synced back to sglang on next generation
#
# Prerequisites:
#   - prepare_self_distill_data.py must have been run to create SD prompts
#     parquet from teacher data (same data format as self-distillation)
#
# Usage:
#   MODEL_PATH=/path/to/model \
#   SD_PROMPTS_PATH=/path/to/self_distill_prompts.parquet \
#   bash train_opsd.sh
# =============================================================================

set -xeo pipefail

ulimit -n 65535

# NOTE: don't enable PYTORCH_CUDA_ALLOC_CONF=expandable_segments here.
# sglang's torch_memory_saver explicitly refuses to initialise when the
# expandable_segments allocator is selected:
#   RuntimeError: TorchMemorySaver is disabled for the current process
#   because expandable_segments is not supported yet.
# which crashes model_runner.load_model during rollout init (observed
# in tu50/tu100 first retries, Flyte f486fd4845df14f5b930 /
# f782dcc0ede2e4c1babf). Until torch_memory_saver supports the new
# allocator, address the reverse-KL / JSD OOMs by lowering JSD/KL
# chunk_size in opsd_worker.py instead.

# =============================================================================
# Environment setup
# =============================================================================
# Default paths — override via environment variables as needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SD_SRC="${WORKSPACE_ROOT}/src"

# Use the image's installed verl as-is. Our dual-path math_verify scorer is a
# standalone module loaded via custom_reward_function.path, not an in-tree verl
# patch.
export PYTHONPATH="${SD_SRC}:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Auto-configure MLflow tracking when running on the Flyte/Kingkong cluster.
# Sets MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME / MLFLOW_RUN_CONTEXT from
# Flyte execution env vars so HuggingFace Trainer (and any other downstream
# code that respects MLFLOW_*) logs to go/mlflowui automatically. Becomes a
# no-op if FLYTE_INTERNAL_EXECUTION_PROJECT is unset (local invocations).
source "${SCRIPT_DIR}/setup_mlflow_hf.sh"


# =============================================================================
# Required environment variables
# =============================================================================
MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
SD_PROMPTS_PATH=${SD_PROMPTS_PATH:?SD_PROMPTS_PATH environment variable is required}
# Optional: reward function file for generation-based validation (_validate).
# Defaults to the standalone dual-path math_verify scorer.
REWARD_FN_PATH=${REWARD_FN_PATH:-${SD_SRC}/rewards/dual_path_math_verify.py}
# Optional: RL-format val dataset for generation-based validation
RL_VAL_FILES=${RL_VAL_FILES:-}
# Optional: prompt template key for process_eval_data.py (e.g. "length_prune_teacher")
# If unset, uses default student template ("think_answer")
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-}

# =============================================================================
# OPSD hyperparameters
# =============================================================================
# JSD interpolation: beta=0.5 gives symmetric JSD
OPSD_BETA=${OPSD_BETA:-0.5}

# Loss type: "jsd" or "reverse_kl"
OPSD_LOSS_TYPE=${OPSD_LOSS_TYPE:-reverse_kl}

# Memory-efficient JSD via logsumexp + progressive teacher freeing
USE_LIGER=${USE_LIGER:-false}

# Structure check: set to false for length pruning (no <think> requirement)
CHECK_STRUCTURE=${CHECK_STRUCTURE:-true}

# Teacher update frequency: hard-copy student weights to teacher every N steps
# 0 = never update (teacher stays frozen at init weights)
TEACHER_UPDATE_FREQ=${TEACHER_UPDATE_FREQ:-0}

# Generation parameters (student rollout from question-only prompt)
SD_TEMPERATURE=${SD_TEMPERATURE:-0.7}
SD_TOP_P=${SD_TOP_P:-0.95}
SD_MAX_TOKENS=${SD_MAX_TOKENS:-8192}
SFT_MAX_LENGTH=${SFT_MAX_LENGTH:-16000}

# Validation generation max tokens (defaults to SD_MAX_TOKENS if not set)
VAL_MAX_TOKENS=${VAL_MAX_TOKENS:-${SD_MAX_TOKENS}}

# =============================================================================
# Training hyperparameters
# =============================================================================
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
# Hard step cap (overrides epoch-based length when set). null/empty = unlimited within epochs.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-}
# Optional: run generation-based val before the first training step. Set to
# "false" for quick smoke tests so we skip the 30-min val_before_train round.
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_FREQ=${SAVE_FREQ:-9999}

# =============================================================================
# Infrastructure
# =============================================================================
N_GPUS=${N_GPUS:-8}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
ULYSSES_SP_SIZE=${ULYSSES_SP_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-${SD_MAX_TOKENS}}

# =============================================================================
# Experiment tracking & output directories
# =============================================================================
PROJECT_NAME=${PROJECT_NAME:-opsd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opsd_hybrid}

# Auto-derive model name from MODEL_PATH (e.g. ".../Qwen3-8B/abc123" -> "qwen3-8b")
MODEL_NAME=$(basename "$(dirname "${MODEL_PATH}")" | tr '[:upper:]' '[:lower:]')
# Prepend model name if not already present in experiment name
if [[ "${EXPERIMENT_NAME}" != *"${MODEL_NAME}"* ]]; then
    EXPERIMENT_NAME="${MODEL_NAME}_${EXPERIMENT_NAME}"
fi
# Append teacher_update_freq if > 0
if [ "${TEACHER_UPDATE_FREQ}" -gt 0 ] 2>/dev/null; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_tu${TEACHER_UPDATE_FREQ}"
fi

# Output directory for checkpoints and detailed logs
NFS_OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXPERIMENT_NAME}}"
mkdir -p "${NFS_OUTPUT_DIR}"

CHECKPOINT_DIR="${NFS_OUTPUT_DIR}/checkpoints"
DETAILED_LOG_DIR="${NFS_OUTPUT_DIR}/detailed_logs"
mkdir -p "${CHECKPOINT_DIR}" "${DETAILED_LOG_DIR}"

echo "============================================="
echo "  OPSD (On-Policy Self-Distillation) Training"
echo "============================================="
echo "  Model:           ${MODEL_PATH}"
echo "  SD Prompts:      ${SD_PROMPTS_PATH}"
echo "  OPSD Beta:       ${OPSD_BETA}"
echo "  Loss type:       ${OPSD_LOSS_TYPE}"
echo "  Temperature:     ${SD_TEMPERATURE}"
echo "  Top-p:           ${SD_TOP_P}"
echo "  Epochs:          ${TOTAL_EPOCHS}"
echo "  Batch size:      ${TRAIN_BATCH_SIZE}"
echo "  Micro batch:     ${MICRO_BATCH_SIZE}"
echo "  LR:              ${LEARNING_RATE}"
echo "  GPUs:            ${N_GPUS} (TP=${TP_SIZE}, SP=${ULYSSES_SP_SIZE})"
echo "  Prompt len:      ${MAX_PROMPT_LENGTH}"
echo "  Response len:    ${MAX_RESPONSE_LENGTH} (student rollout)"
echo "  Val max tokens:  ${VAL_MAX_TOKENS} (validation generation)"
echo "  SFT max len:     ${SFT_MAX_LENGTH}"
echo "  Use liger:       ${USE_LIGER}"
echo "  Check struct:    ${CHECK_STRUCTURE}"
echo "  Teacher update:  every ${TEACHER_UPDATE_FREQ} steps (0=frozen)"
echo "  Reward fn:       ${REWARD_FN_PATH:-<none>}"
echo "  RL val files:    ${RL_VAL_FILES:-<none>}"
echo "  Prompt template: ${PROMPT_TEMPLATE:-<default: think_answer>}"
echo "  NFS output:      ${NFS_OUTPUT_DIR}"
echo "  Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "  Log dir:         ${DETAILED_LOG_DIR}"
echo "============================================="

# =============================================================================
# Process validation data on-cluster (ensures consistent library versions)
# =============================================================================
DATA_DIR="${WORKSPACE_ROOT}/data"
PROCESSED_DIR="${DATA_DIR}/processed"

if [ -d "${DATA_DIR}" ]; then
    echo "Processing validation data on-cluster..."
    python3 "${SD_SRC}/data/process_eval_data.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PROCESSED_DIR}" \
        ${PROMPT_TEMPLATE:+--prompt_template "${PROMPT_TEMPLATE}"}
    echo "Data processing complete."
fi

# If RL_VAL_FILES not set externally, auto-detect processed val datasets
if [ -z "${RL_VAL_FILES}" ] && [ -f "${PROCESSED_DIR}/val_math500.parquet" ]; then
    VAL_FILE_LIST="'${PROCESSED_DIR}/val_math500.parquet'"
    if [ -f "${PROCESSED_DIR}/val_aime24.parquet" ]; then
        VAL_FILE_LIST="${VAL_FILE_LIST}, '${PROCESSED_DIR}/val_aime24.parquet'"
    fi
    if [ -f "${PROCESSED_DIR}/val_aime25.parquet" ]; then
        VAL_FILE_LIST="${VAL_FILE_LIST}, '${PROCESSED_DIR}/val_aime25.parquet'"
    fi
    RL_VAL_FILES="[${VAL_FILE_LIST}]"
    echo "Auto-set RL_VAL_FILES=${RL_VAL_FILES}"
fi

# =============================================================================
# Launch OPSD training
# =============================================================================
python3 -m self_distill_hybrid.main_opsd \
    --config-path "${SD_SRC}/self_distill_hybrid/config" \
    --config-name opsd_trainer \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_liger="${USE_LIGER}" \
    \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.min_lr_ratio=null \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature="${SD_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${SD_TOP_P}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    \
    data.train_files="${SD_PROMPTS_PATH}" \
    ${RL_VAL_FILES:+data.val_files="${RL_VAL_FILES}"} \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=true \
    data.shuffle=true \
    data.dataloader_num_workers=4 \
    \
    opsd.beta="${OPSD_BETA}" \
    opsd.loss_type="${OPSD_LOSS_TYPE}" \
    opsd.sft_max_length="${SFT_MAX_LENGTH}" \
    opsd.check_structure="${CHECK_STRUCTURE}" \
    opsd.sd_max_tokens="${SD_MAX_TOKENS}" \
    opsd.val_max_tokens="${VAL_MAX_TOKENS}" \
    ${TEST_FREQ:+opsd.test_freq="${TEST_FREQ}"} \
    opsd.teacher_update_freq="${TEACHER_UPDATE_FREQ}" \
    \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"} \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.logger='["console", "mlflow"]' \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    opsd.detailed_log_dir="${DETAILED_LOG_DIR}" \
    ${TRAIN_MAX_SAMPLES:+data.train_max_samples=${TRAIN_MAX_SAMPLES}} \
    ${REWARD_FN_PATH:+reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score} \
    "$@"

echo ""
echo "============================================="
echo "  OPSD Training complete!"
echo "============================================="
