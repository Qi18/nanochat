#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${EXPERIMENT_DIR}/config/sft-eval.env"
export PYTHONPATH="${NANOCHAT_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_dir="${NANOCHAT_BASE_DIR}/chatsft_checkpoints/${SFT_EVAL_MODEL_TAG}"
checkpoint_step="$(printf '%06d' "${SFT_EVAL_STEP}")"
test -f "${checkpoint_dir}/model_${checkpoint_step}.pt"
test -f "${checkpoint_dir}/meta_${checkpoint_step}.json"
for manifest in \
    "cais--mmlu/all/test" \
    "openai--gsm8k/main/test"; do
    test -f "${NANOCHAT_BASE_DIR}/task_data/${manifest}/manifest.json"
done

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    "${EXPERIMENT_DIR}/scripts/run_chat_eval_sdpa.py" \
    -- \
    "--source=${SFT_EVAL_SOURCE}" \
    "--task-name=${SFT_EVAL_TASKS}" \
    "--model-tag=${SFT_EVAL_MODEL_TAG}" \
    "--step=${SFT_EVAL_STEP}" \
    "--batch-size=${SFT_EVAL_BATCH_SIZE}" \
    "--num-samples=${SFT_EVAL_NUM_SAMPLES}" \
    "--max-new-tokens=${SFT_EVAL_MAX_NEW_TOKENS}" \
    "--temperature=${SFT_EVAL_TEMPERATURE}" \
    "--top-k=${SFT_EVAL_TOP_K}" \
    2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/sft-eval.log"
