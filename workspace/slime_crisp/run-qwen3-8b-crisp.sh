#!/bin/bash
# =============================================================================
# CRISP on slime — Qwen3-8B, milestone 1
#
# Teacher = the SAME Qwen3-8B conditioned on a conciseness prompt, served on a
# dedicated GPU; refreshed from the student every M=50 rollouts (progressive
# compression). Student trains via slime's sglang-mode OPD (sampled-token
# reverse KL as an advantage penalty), reward = 0 (pure distillation).
#
# Paper recipe (Table 2 / §5.1): batch 32, lr 1e-6, temp 1.0, top-p 1.0,
# 8192-token rollouts, single rollout per prompt, ~100 steps, M=50.
#
# GPU layout (8 GPUs): actor 4 | rollout 3 | teacher 1 (GPU 7).
# Uses synchronous train.py — REQUIRED for correct teacher-refresh ordering.
#
# KL estimator (CRISP_KL_MODE):
#   sampled (default) — milestone 1: sampled-token reverse KL via slime's OPD
#                       advantage penalty (--use-opd).
#   full              — milestone 2 (FULL_KL_PLAN.md): bucketed full-vocab
#                       reverse KL over teacher top-K as a custom Megatron
#                       loss. Requires the patched slime (crisp branch).
#
# Prereqs:
#   1. HF checkpoint at $HF_CKPT, converted Megatron ckpt at $MCORE_CKPT
#      (tools/convert_hf_to_torch_dist.py, see slime OPD example README)
#   2. Data: python prepare_crisp_slime_data.py --input-parquet ... --output $PROMPT_DATA
#   3. This repo's workspace/ dir on PYTHONPATH (for slime_crisp.*)
# =============================================================================

set -ex

# ---- Paths (override via env) ----
HF_CKPT=${HF_CKPT:-/root/Qwen3-8B}
MCORE_CKPT=${MCORE_CKPT:-/root/Qwen3-8B_torch_dist}
SAVE_DIR=${SAVE_DIR:-/root/Qwen3-8B_crisp_slime}
TEACHER_HF_PATH=${TEACHER_HF_PATH:-/root/crisp_teacher_hf}   # must match crisp_config.yaml
PROMPT_DATA=${PROMPT_DATA:-/root/data/dapo_math_crisp.jsonl}
SLIME_DIR=${SLIME_DIR:-/root/slime}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
CRISP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../workspace/slime_crisp
WORKSPACE_DIR="$(dirname "${CRISP_DIR}")"                          # .../workspace

# ---- Teacher server (same base checkpoint, dedicated GPU) ----
TEACHER_IP=${TEACHER_IP:-127.0.0.1}
TEACHER_PORT=${TEACHER_PORT:-13141}
LOG_FILE="/tmp/sglang_teacher_$(date +%s).log"

CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
    --model-path ${HF_CKPT} \
    --host 0.0.0.0 \
    --port ${TEACHER_PORT} \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.7 \
    > "${LOG_FILE}" 2>&1 &

echo "Waiting for teacher server..."
until curl -sf http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null; do
    tail -n 5 "${LOG_FILE}" || true
    sleep 5
done
echo "Teacher up at ${TEACHER_IP}:${TEACHER_PORT}"

export PYTHONUNBUFFERED=1
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --load ${MCORE_CKPT}
   --save ${SAVE_DIR}
   --save-interval 50                      # == crisp_teacher_update_interval (M)
   --save-hf ${TEACHER_HF_PATH}            # fixed path: each save overwrites; teacher reloads it
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   # Pin Qwen3 thinking mode (mirrors verl opsd_trainer.yaml; guards against
   # tokenizer default drift). Also forwarded to the teacher prompt in crisp_opd.
   --apply-chat-template-kwargs '{"enable_thinking": true}'
   --rollout-shuffle
   --num-rollout 100                       # paper: step 100 is the sweet spot
   --rollout-batch-size 32
   --n-samples-per-prompt 1                # CRISP: single rollout per prompt
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --rollout-top-p 1.0
   --global-batch-size 32                  # 1 optimizer step per rollout
)

CRISP_KL_MODE=${CRISP_KL_MODE:-sampled}

if [ "${CRISP_KL_MODE}" = "full" ]; then
    CRISP_CONFIG=${CRISP_DIR}/crisp_config_full_kl.yaml
    DISTILL_ARGS=(
       # Distribution-level loss replaces the PG/OPD pipeline entirely.
       --loss-type custom_loss
       --custom-loss-function-path slime_crisp.crisp_full_kl_loss.full_kl_loss_function
       # Skip advantages AND the old-log-prob forward pass — the custom loss
       # needs only the training forward.
       --disable-compute-advantages-and-returns
       # Recompute the loss fn in backward: frees the per-chunk softmax graph,
       # bounding loss memory to the [T, V] logits already resident.
       --recompute-loss-function
       --advantage-estimator grpo           # unused (advantages disabled); schema only
       --entropy-coef 0.00
       --calculate-per-token-loss           # global token mean == verl normalization
    )
else
    CRISP_CONFIG=${CRISP_DIR}/crisp_config.yaml
    DISTILL_ARGS=(
       --advantage-estimator grpo           # degenerate at n=1 + reward 0: pure OPD penalty
       --use-opd
       --opd-type sglang
       --opd-kl-coef 1.0
       --entropy-coef 0.00
       # Global token-mean loss reduction: closest match to verl OPSD's
       # kl_sum / n_tokens normalization (default slime reduction is per-rollout mean).
       --calculate-per-token-loss
    )
fi

CRISP_ARGS=(
   --custom-rm-path slime_crisp.crisp_opd.reward_func
   --custom-reward-post-process-path slime_crisp.crisp_opd.post_process_rewards
   --rollout-function-path slime_crisp.crisp_opd.generate_rollout
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
   --custom-config-path ${CRISP_CONFIG}
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   # Match verl's AdamW defaults (train_opsd.sh leaves these unset).
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 7 --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${MEGATRON_DIR}:${WORKSPACE_DIR}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
     }
   }" \
   -- python3 ${SLIME_DIR}/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 3 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTILL_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CRISP_ARGS[@]}

# ---- cleanup ----
pkill -9 sglang || true
sleep 3
ray stop --force || true
