"""Upload SFT results measured with the Base BPB/CORE protocol to SwanLab."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import swanlab


def parse_bpb(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    out = {}
    for split in ("train", "val"):
        match = re.search(rf"^{split} bpb: ([0-9.]+)$", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Missing {split} BPB")
        out[split] = float(match.group(1))
    return out


def parse_core(path: Path) -> tuple[dict[str, dict[str, float]], float]:
    tasks: dict[str, dict[str, float]] = {}
    core = None
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            task = row[0].strip()
            if task == "Task":
                continue
            accuracy = row[1].strip()
            centered = row[2].strip()
            if task == "CORE":
                core = float(centered)
            else:
                tasks[task] = {
                    "accuracy": float(accuracy),
                    "centered": float(centered),
                }
    if core is None or len(tasks) != 22:
        raise RuntimeError(f"Incomplete CORE CSV: tasks={len(tasks)}, core={core}")
    return tasks, core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    bpb = parse_bpb(args.log)
    tasks, core = parse_core(args.csv)
    run = swanlab.init(
        project=args.project,
        name=args.run_name,
        group=args.group,
        job_type="sft-base-protocol-eval",
        tags=["l20", "d24", "bf16", "sft", "eval", "full", "base-protocol", "fa3"],
        mode="online",
        log_dir=str(args.log_dir),
        config={
            "source": "sft",
            "model_tag": args.model_tag,
            "checkpoint_step": args.step,
            "world_size": 8,
            "dtype": "bfloat16",
            "attention_backend": "flash_attention_3",
            "eval_modes": ["sample", "bpb", "core"],
            "device_batch_size": 2,
            "split_tokens": 20971520,
            "max_per_task": -1,
        },
    )
    payload = {
        "eval/train_bpb": bpb["train"],
        "eval/val_bpb": bpb["val"],
        "eval/CORE": core,
    }
    for task, values in tasks.items():
        payload[f"eval/core/{task}/accuracy"] = values["accuracy"]
        payload[f"eval/core/{task}/centered"] = values["centered"]
    run.log(payload, step=args.step)
    run.finish()


if __name__ == "__main__":
    main()
