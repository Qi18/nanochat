#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${EXPERIMENT_DIR}/config/sft-v2.env"
export PYTHONPATH="${NANOCHAT_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_dir="${NANOCHAT_BASE_DIR}/base_checkpoints/${SFT_V2_SOURCE_MODEL_TAG}"
test -f "${checkpoint_dir}/model_$(printf '%06d' "${SFT_V2_SOURCE_MODEL_STEP}").pt"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    "${EXPERIMENT_DIR}/scripts/run_chat_sft_sdpa.py" \
    -- \
    "--model-tag=${SFT_V2_SOURCE_MODEL_TAG}" \
    "--model-step=${SFT_V2_SOURCE_MODEL_STEP}" \
    "--output-model-tag=${SFT_V2_OUTPUT_MODEL_TAG}-probe" \
    "--load-optimizer=${SFT_V2_LOAD_OPTIMIZER}" \
    --num-iterations=3 \
    "--max-seq-len=${SFT_V2_MAX_SEQ_LEN}" \
    "--device-batch-size=${SFT_V2_DEVICE_BATCH_SIZE}" \
    "--total-batch-size=${SFT_V2_TOTAL_BATCH_SIZE}" \
    "--init-lr-frac=${SFT_V2_INIT_LR_FRAC}" \
    "--warmup-ratio=${SFT_V2_WARMUP_RATIO}" \
    "--warmdown-ratio=${SFT_V2_WARMDOWN_RATIO}" \
    "--final-lr-frac=${SFT_V2_FINAL_LR_FRAC}" \
    --eval-every=-1 \
    --eval-tokens=524288 \
    --base-eval-every=1 \
    --base-eval-tokens=524288 \
    --chatcore-every=-1 \
    --checkpoint-every=-1 \
    --save-intermediate-optimizer=0 \
    "--mmlu-epochs=${SFT_V2_MMLU_EPOCHS}" \
    "--gsm8k-epochs=${SFT_V2_GSM8K_EPOCHS}" \
    "--run=${EXPERIMENT_ID}-sft-v2-probe" \
    --swanlab-mode=online \
    "--swanlab-project=${NANOCHAT_SWANLAB_PROJECT}" \
    "--swanlab-group=${NANOCHAT_SWANLAB_GROUP}" \
    "--swanlab-tags=${NANOCHAT_SWANLAB_TAGS},sft,v2,probe,lr02,fresh-optimizer,sdpa" \
    2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/sft-v2-probe.log"
