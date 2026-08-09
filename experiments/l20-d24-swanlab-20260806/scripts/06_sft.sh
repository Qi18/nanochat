#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${EXPERIMENT_DIR}/config/sft.env"
export PYTHONPATH="${NANOCHAT_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_dir="${NANOCHAT_BASE_DIR}/base_checkpoints/${SFT_MODEL_TAG}"
test -f "${checkpoint_dir}/model_$(printf '%06d' "${SFT_MODEL_STEP}").pt"
for rank in $(seq 0 7); do
    test -f "${checkpoint_dir}/optim_$(printf '%06d' "${SFT_MODEL_STEP}")_rank${rank}.pt"
done
for manifest in \
    "HuggingFaceTB--smol-smoltalk/default/train" \
    "HuggingFaceTB--smol-smoltalk/default/test" \
    "cais--mmlu/all/auxiliary_train" \
    "cais--mmlu/all/test" \
    "openai--gsm8k/main/train" \
    "openai--gsm8k/main/test"; do
    test -f "${NANOCHAT_BASE_DIR}/task_data/${manifest}/manifest.json"
done

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    "${EXPERIMENT_DIR}/scripts/run_chat_sft_sdpa.py" \
    -- \
    "--model-tag=${SFT_MODEL_TAG}" \
    "--model-step=${SFT_MODEL_STEP}" \
    "--load-optimizer=${SFT_LOAD_OPTIMIZER}" \
    "--num-iterations=${SFT_NUM_ITERATIONS}" \
    "--max-seq-len=${SFT_MAX_SEQ_LEN}" \
    "--device-batch-size=${SFT_DEVICE_BATCH_SIZE}" \
    "--total-batch-size=${SFT_TOTAL_BATCH_SIZE}" \
    "--eval-every=${SFT_EVAL_EVERY}" \
    "--eval-tokens=${SFT_EVAL_TOKENS}" \
    "--chatcore-every=${SFT_CHATCORE_EVERY}" \
    "--chatcore-max-cat=${SFT_CHATCORE_MAX_CAT}" \
    "--chatcore-max-sample=${SFT_CHATCORE_MAX_SAMPLE}" \
    "--mmlu-epochs=${SFT_MMLU_EPOCHS}" \
    "--gsm8k-epochs=${SFT_GSM8K_EPOCHS}" \
    "--run=${EXPERIMENT_ID}-sft" \
    "--swanlab-mode=offline" \
    "--swanlab-project=${NANOCHAT_SWANLAB_PROJECT}" \
    "--swanlab-group=${NANOCHAT_SWANLAB_GROUP}" \
    "--swanlab-tags=${NANOCHAT_SWANLAB_TAGS},sft,sdpa" \
    2>&1 | tee "${EXPERIMENT_RUNTIME_DIR}/sft.log"
