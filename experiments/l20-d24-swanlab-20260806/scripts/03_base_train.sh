#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${EXPERIMENT_DIR}/config/base.env"
source "${EXPERIMENT_DIR}/config/smoke.env"

require_clean_repo

test -f "${NANOCHAT_BASE_DIR}/base_checkpoints/${SMOKE_MODEL_TAG}/model_000100.pt"

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- +    "--depth=${BASE_DEPTH}" +    "--target-param-data-ratio=${BASE_TARGET_PARAM_DATA_RATIO}" +    "--max-seq-len=${BASE_MAX_SEQ_LEN}" +    "--window-pattern=${BASE_WINDOW_PATTERN}" +    "--device-batch-size=${BASE_DEVICE_BATCH_SIZE}" +    "--total-batch-size=${BASE_TOTAL_BATCH_SIZE}" +    "--eval-every=${BASE_EVAL_EVERY}" +    "--eval-tokens=${BASE_EVAL_TOKENS}" +    "--core-metric-every=${BASE_CORE_METRIC_EVERY}" +    "--core-metric-max-per-task=${BASE_CORE_METRIC_MAX_PER_TASK}" +    "--sample-every=${BASE_SAMPLE_EVERY}" +    "--save-every=${BASE_SAVE_EVERY}" +    "--model-tag=${BASE_MODEL_TAG}" +    "--run=${EXPERIMENT_ID}-base" +    "--swanlab-mode=${NANOCHAT_SWANLAB_MODE}" +    "--swanlab-project=${NANOCHAT_SWANLAB_PROJECT}" +    "--swanlab-group=${NANOCHAT_SWANLAB_GROUP}" +    "--swanlab-tags=${NANOCHAT_SWANLAB_TAGS},base" +    2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/base-train.log"
