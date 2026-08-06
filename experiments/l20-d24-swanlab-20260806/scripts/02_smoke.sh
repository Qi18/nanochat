#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${EXPERIMENT_DIR}/config/smoke.env"

require_clean_repo

COMMON_ARGS=(
    "--depth=${SMOKE_DEPTH}"
    "--max-seq-len=${SMOKE_MAX_SEQ_LEN}"
    "--window-pattern=L"
    "--device-batch-size=${SMOKE_DEVICE_BATCH_SIZE}"
    "--total-batch-size=${SMOKE_TOTAL_BATCH_SIZE}"
    "--eval-every=${SMOKE_EVAL_EVERY}"
    "--eval-tokens=${SMOKE_EVAL_TOKENS}"
    "--core-metric-every=-1"
    "--sample-every=-1"
    "--save-every=${SMOKE_SAVE_EVERY}"
    "--model-tag=${SMOKE_MODEL_TAG}"
    "--swanlab-mode=${NANOCHAT_SWANLAB_MODE}"
    "--swanlab-project=${NANOCHAT_SWANLAB_PROJECT}"
    "--swanlab-group=${NANOCHAT_SWANLAB_GROUP}"
    "--swanlab-tags=${NANOCHAT_SWANLAB_TAGS},smoke"
)

smoke_a_command=(
    python -m torch.distributed.run
    --standalone
    --nproc_per_node=8
    -m scripts.base_train
    --
    "${COMMON_ARGS[@]}"
    "--num-iterations=${SMOKE_FIRST_STOP}"
    "--run=${EXPERIMENT_ID}-smoke-a"
)
"${smoke_a_command[@]}" 2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/smoke-a.log"

test -f "${NANOCHAT_BASE_DIR}/base_checkpoints/${SMOKE_MODEL_TAG}/model_000050.pt"

smoke_resume_command=(
    python -m torch.distributed.run
    --standalone
    --nproc_per_node=8
    -m scripts.base_train
    --
    "${COMMON_ARGS[@]}"
    "--num-iterations=${SMOKE_FINAL_STOP}"
    "--resume-from-step=${SMOKE_FIRST_STOP}"
    "--run=${EXPERIMENT_ID}-smoke-resume"
)
"${smoke_resume_command[@]}" 2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/smoke-resume.log"

test -f "${NANOCHAT_BASE_DIR}/base_checkpoints/${SMOKE_MODEL_TAG}/model_000100.pt"
echo "8-GPU smoke and checkpoint resume complete"
