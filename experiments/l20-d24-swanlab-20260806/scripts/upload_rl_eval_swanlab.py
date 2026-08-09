"""Upload RL Chat Eval and Base protocol metrics to one SwanLab run."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import swanlab


TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")


def parse_chat(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    out: dict[str, float] = {}
    for task in TASKS:
        match = re.search(rf"^{re.escape(task)} accuracy: ([0-9.]+)%$", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Missing {task} in {path}")
        out[task] = float(match.group(1)) / 100.0
    match = re.search(r"^ChatCORE metric: ([0-9.]+)$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Missing ChatCORE in {path}")
    out["ChatCORE"] = float(match.group(1))
    return out


def parse_base(log_path: Path, csv_path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]], float]:
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    bpb: dict[str, float] = {}
    for split in ("train", "val"):
        match = re.search(rf"^{split} bpb: ([0-9.]+)$", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Missing {split} BPB in {log_path}")
        bpb[split] = float(match.group(1))
    tasks: dict[str, dict[str, float]] = {}
    core = None
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            task = row[0].strip()
            if task in ("Task", "CORE"):
                if task == "CORE":
                    core = float(row[2].strip())
                continue
            tasks[task] = {"accuracy": float(row[1].strip()), "centered": float(row[2].strip())}
    if core is None or len(tasks) != 22:
        raise RuntimeError(f"Incomplete CORE CSV {csv_path}: tasks={len(tasks)}, core={core}")
    return bpb, tasks, core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", nargs=2, type=Path, metavar=("STEP420_LOG", "STEP466_LOG"), required=True)
    parser.add_argument("--base", nargs=2, type=Path, metavar=("STEP420_LOG", "STEP466_LOG"), required=True)
    parser.add_argument("--base-csv", nargs=2, type=Path, metavar=("STEP420_CSV", "STEP466_CSV"), required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    args = parser.parse_args()

    run = swanlab.init(
        project=args.project,
        name=args.run_name,
        group=args.group,
        job_type="rl-eval",
        tags=["l20", "d24", "bf16", "rl", "eval", "full", "sdpa", "base-protocol"],
        mode="online",
        log_dir=str(args.log_dir),
        config={
            "source": "rl",
            "model_tag": args.model_tag,
            "checkpoint_steps": [420, 466],
            "world_size": 8,
            "dtype": "bfloat16",
            "chat_attention_backend": "pytorch_sdpa",
            "base_attention_backend": "flash_attention_3",
            "chat_num_samples": 1,
            "chat_max_new_tokens": 512,
            "chat_temperature": 0.0,
            "base_eval_modes": ["sample", "bpb", "core"],
            "base_device_batch_size": 2,
            "base_split_tokens": 20971520,
        },
    )
    for step, chat_path, base_path, csv_path in zip((420, 466), args.chat, args.base, args.base_csv):
        chat = parse_chat(chat_path)
        bpb, tasks, core = parse_base(base_path, csv_path)
        payload = {f"step_{step}/chat/{task}": value for task, value in chat.items()}
        payload.update({f"step_{step}/base/{name}_bpb": value for name, value in bpb.items()})
        payload[f"step_{step}/base/CORE"] = core
        for task, values in tasks.items():
            payload[f"step_{step}/base/core/{task}/accuracy"] = values["accuracy"]
            payload[f"step_{step}/base/core/{task}/centered"] = values["centered"]
        run.log(payload, step=step)
    run.finish()


if __name__ == "__main__":
    main()
