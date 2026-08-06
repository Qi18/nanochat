#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_clean_repo

LOG_FILE="${EXPERIMENT_RUNTIME_DIR}/preflight.log"
PROVENANCE_FILE="${EXPERIMENT_DIR}/config/provenance.json"

{
    date --iso-8601=seconds
    hostname
    git status --short --branch
    git rev-parse HEAD
    git remote -v
    python --version
    python -c 'import torch, swanlab, wandb, rustbpe, kernels; print("torch", torch.__version__, "cuda", torch.version.cuda); print("swanlab", swanlab.__version__); print("wandb", wandb.__version__); x=torch.ones(1024, device="cuda"); print("gpu_count", torch.cuda.device_count(), "cuda_sum", x.sum().item())'
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    df -h /data /dev/shm
    python -m torch.distributed.run --standalone --nproc_per_node=8 "${SCRIPT_DIR}/nccl_smoke.py"
    curl -L -sS -I -o /dev/null --connect-timeout 10 --max-time 30 -w 'dataset_mirror_http=%{http_code}\n' "${NANOCHAT_DATASET_BASE_URL}/shard_00000.parquet"
    git ls-remote origin "refs/heads/${EXPERIMENT_BRANCH}"
} 2>&1 | tee "${LOG_FILE}"

python - "${PROVENANCE_FILE}" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import swanlab
import torch
import wandb


def run(*args):
    return subprocess.check_output(args, text=True).strip()


out = sys.argv[1]
gpu_lines = run(
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,driver_version",
    "--format=csv,noheader",
).splitlines()
payload = {
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "git_branch": run("git", "branch", "--show-current"),
    "git_commit": run("git", "rev-parse", "HEAD"),
    "git_dirty": bool(run("git", "status", "--porcelain")),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_compiled_cuda": torch.version.cuda,
    "swanlab": swanlab.__version__,
    "wandb": wandb.__version__,
    "compute_dtype": os.environ["NANOCHAT_DTYPE"],
    "dataset_base_url": os.environ["NANOCHAT_DATASET_BASE_URL"],
    "nanochat_base_dir": os.environ["NANOCHAT_BASE_DIR"],
    "gpus": gpu_lines,
    "cuda_device_count": torch.cuda.device_count(),
    "nccl_available": torch.distributed.is_nccl_available(),
}
encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode()
payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(out)
PY

echo "preflight complete: ${LOG_FILE}"
