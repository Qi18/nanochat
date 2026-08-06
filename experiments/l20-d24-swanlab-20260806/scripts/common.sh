#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${EXPERIMENT_DIR}/config/environment.env"

export PATH="${NANOCHAT_VENV}/bin:${PATH}"

mkdir -p "${NANOCHAT_BASE_DIR}" "${HF_HOME}" "${SWANLAB_LOGDIR}" "${EXPERIMENT_RUNTIME_DIR}"
cd "${NANOCHAT_REPO}"

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "${EXPERIMENT_BRANCH}" ]]; then
    echo "ERROR: expected branch ${EXPERIMENT_BRANCH}, got ${current_branch}" >&2
    exit 1
fi

require_clean_repo() {
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERROR: repository must be clean before training" >&2
        git status --short >&2
        exit 1
    fi
}
