#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${EXPERIMENT_DIR}/config/sft-base-protocol-eval.env"
export PYTHONPATH="${NANOCHAT_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_dir="${NANOCHAT_BASE_DIR}/chatsft_checkpoints/${SFT_BASE_EVAL_MODEL_TAG}"
checkpoint_step="$(printf '%06d' "${SFT_BASE_EVAL_STEP}")"
test -f "${checkpoint_dir}/model_${checkpoint_step}.pt"
test -f "${checkpoint_dir}/meta_${checkpoint_step}.json"
test -d "${NANOCHAT_BASE_DIR}/eval_bundle"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    -m scripts.base_eval \
    -- \
    --source=sft \
    "--eval=${SFT_BASE_EVAL_MODES}" \
    "--model-tag=${SFT_BASE_EVAL_MODEL_TAG}" \
    "--step=${SFT_BASE_EVAL_STEP}" \
    "--device-batch-size=${SFT_BASE_EVAL_DEVICE_BATCH_SIZE}" \
    "--split-tokens=${SFT_BASE_EVAL_SPLIT_TOKENS}" \
    "--max-per-task=${SFT_BASE_EVAL_MAX_PER_TASK}" \
    2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/sft-base-protocol-eval.log"
