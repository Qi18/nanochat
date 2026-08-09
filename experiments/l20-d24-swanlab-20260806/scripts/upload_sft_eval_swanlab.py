"""Parse the full SFT evaluation log and upload its metrics to SwanLab."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import swanlab


TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")
EXPECTED_TOTALS = {
    "ARC-Easy": 2376,
    "ARC-Challenge": 1172,
    "MMLU": 14042,
    "GSM8K": 1319,
    "HumanEval": 164,
}


def parse_log(path: Path) -> tuple[dict[str, float], float]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    metrics: dict[str, float] = {}
    for task in TASKS:
        match = re.search(rf"^{re.escape(task)} accuracy: ([0-9.]+)%$", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Missing completed metric for {task}")
        metrics[task] = float(match.group(1)) / 100.0
    match = re.search(r"^ChatCORE metric: ([0-9.]+)$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError("Missing completed ChatCORE metric")
    return metrics, float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    metrics, chatcore = parse_log(args.log)
    run = swanlab.init(
        project=args.project,
        name=args.run_name,
        group=args.group,
        job_type="sft-eval",
        tags=["l20", "d24", "bf16", "sft", "eval", "full", "sdpa"],
        mode="online",
        log_dir=str(args.log_dir),
        config={
            "source": "sft",
            "model_tag": args.model_tag,
            "checkpoint_step": args.step,
            "world_size": 8,
            "dtype": "bfloat16",
            "attention_backend": "pytorch_sdpa",
            "batch_size": 8,
            "num_samples": 1,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_k": 50,
            "max_problems": None,
            "task_totals": EXPECTED_TOTALS,
        },
    )
    payload = {f"eval/{task}": value for task, value in metrics.items()}
    payload["eval/ChatCORE"] = chatcore
    run.log(payload, step=args.step)
    run.finish()


if __name__ == "__main__":
    main()
