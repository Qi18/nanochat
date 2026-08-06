#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_clean_repo

INITIAL_LOG="${EXPERIMENT_RUNTIME_DIR}/dataset-initial.log"
FULL_LOG="${EXPERIMENT_RUNTIME_DIR}/dataset-full.log"
TOKENIZER_LOG="${EXPERIMENT_RUNTIME_DIR}/tokenizer-train.log"
TOKENIZER_EVAL_LOG="${EXPERIMENT_RUNTIME_DIR}/tokenizer-eval.log"
MANIFEST_FILE="${EXPERIMENT_DIR}/config/data_manifest.json"

python -m nanochat.dataset -n 8 -w 4 2>&1 | tee "${INITIAL_LOG}"

python -m nanochat.dataset -n 170 -w 8 >"${FULL_LOG}" 2>&1 &
download_pid=$!

cleanup() {
    if kill -0 "${download_pid}" 2>/dev/null; then
        kill "${download_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

python -m scripts.tok_train 2>&1 | tee "${TOKENIZER_LOG}"
python -m scripts.tok_eval 2>&1 | tee "${TOKENIZER_EVAL_LOG}"
wait "${download_pid}"
trap - EXIT

python - "${MANIFEST_FILE}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

base = Path(os.environ["NANOCHAT_BASE_DIR"])
data_dir = base / "base_data_climbmix"
tokenizer_dir = base / "tokenizer"
shards = sorted(data_dir.glob("shard_*.parquet"))
entries = [{"name": path.name, "size_bytes": path.stat().st_size} for path in shards]
listing = "\n".join(f"{item['name']}\t{item['size_bytes']}" for item in entries).encode()

tokenizer_files = []
for path in sorted(tokenizer_dir.iterdir()):
    if not path.is_file():
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tokenizer_files.append(
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest}
    )

payload = {
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "dataset_base_url": os.environ["NANOCHAT_DATASET_BASE_URL"],
    "data_dir": str(data_dir),
    "shard_count": len(entries),
    "shard_listing_sha256": hashlib.sha256(listing).hexdigest(),
    "shards": entries,
    "tokenizer_dir": str(tokenizer_dir),
    "tokenizer_files": tokenizer_files,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(sys.argv[1])
PY

echo "data and tokenizer preparation complete"
