#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${EXPERIMENT_DIR}/config/base-eval.env"

checkpoint_dir="${NANOCHAT_BASE_DIR}/base_checkpoints/${BASE_EVAL_MODEL_TAG}"
test -f "${checkpoint_dir}/model_$(printf '%06d' "${BASE_EVAL_STEP}").pt"
test -f "${checkpoint_dir}/meta_$(printf '%06d' "${BASE_EVAL_STEP}").json"
test -d "${NANOCHAT_BASE_DIR}/eval_bundle"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    -m scripts.base_eval \
    -- \
    "--eval=${BASE_EVAL_MODES}" \
    "--model-tag=${BASE_EVAL_MODEL_TAG}" \
    "--step=${BASE_EVAL_STEP}" \
    "--device-batch-size=${BASE_EVAL_DEVICE_BATCH_SIZE}" \
    "--split-tokens=${BASE_EVAL_SPLIT_TOKENS}" \
    "--max-per-task=${BASE_EVAL_MAX_PER_TASK}" \
    2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/base-eval.log"
