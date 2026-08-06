"""Optional SwanLab experiment tracking built on nanochat's W&B logging calls."""

import argparse
import importlib
import os

import wandb

from nanochat.common import DummyWandb, print0


SWANLAB_MODES = ("disabled", "online", "local", "offline")


def dataloader_tracking_metrics(state: dict) -> dict[str, int]:
    """Return scalar-only dataloader metrics accepted by W&B and SwanLab."""
    return {
        "train/epoch": int(state["epoch"]),
        "train/pq_idx": int(state["pq_idx"]),
        "train/rg_idx": int(state["rg_idx"]),
    }


def add_tracking_args(parser: argparse.ArgumentParser) -> None:
    """Add shared W&B/SwanLab options to a training argument parser."""
    parser.add_argument(
        "--run",
        type=str,
        default="dummy",
        help="experiment name ('dummy' disables W&B and SwanLab logging)",
    )
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        default=os.environ.get("NANOCHAT_SWANLAB_MODE", "disabled"),
        choices=SWANLAB_MODES,
        help="SwanLab mode; disabled preserves the upstream W&B-only behavior",
    )
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=os.environ.get("NANOCHAT_SWANLAB_PROJECT", "nanochat-lab"),
        help="SwanLab project used to group base, SFT, and RL runs",
    )
    parser.add_argument(
        "--swanlab-workspace",
        type=str,
        default=os.environ.get("NANOCHAT_SWANLAB_WORKSPACE"),
        help="optional SwanLab workspace/user name",
    )
    parser.add_argument(
        "--swanlab-group",
        type=str,
        default=os.environ.get("NANOCHAT_SWANLAB_GROUP"),
        help="optional SwanLab experiment group, e.g. l20-d24-2026-08",
    )
    parser.add_argument(
        "--swanlab-tags",
        type=str,
        default=os.environ.get("NANOCHAT_SWANLAB_TAGS", ""),
        help="comma-separated SwanLab tags",
    )


def init_experiment_tracking(
    args: argparse.Namespace,
    config: dict,
    *,
    master_process: bool,
    wandb_project: str,
    job_type: str,
):
    """Initialize W&B and optionally mirror its metrics into SwanLab.

    Only DDP rank 0 initializes tracking. SwanLab is imported lazily so the
    default upstream workflow has no additional runtime dependency.
    """
    if args.run == "dummy" or not master_process:
        if master_process and args.swanlab_mode != "disabled":
            print0("SwanLab logging requested but disabled because --run=dummy")
        return DummyWandb()

    if args.swanlab_mode != "disabled":
        try:
            swanlab = importlib.import_module("swanlab")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SwanLab logging requires the tracking extra: "
                "uv sync --extra cpu --extra tracking (or --extra gpu)"
            ) from exc

        tags = [tag.strip() for tag in args.swanlab_tags.split(",") if tag.strip()]
        init_kwargs = {
            "project": args.swanlab_project,
            "experiment_name": args.run,
            "job_type": job_type,
            "config": config,
            "mode": args.swanlab_mode,
        }
        if args.swanlab_workspace:
            init_kwargs["workspace"] = args.swanlab_workspace
        if args.swanlab_group:
            init_kwargs["group"] = args.swanlab_group
        if tags:
            init_kwargs["tags"] = tags

        swanlab.init(**init_kwargs)
        # Keep nanochat's existing wandb.log calls but do not upload to W&B.
        swanlab.sync_wandb(mode=args.swanlab_mode, wandb_run=False)
        print0(
            f"SwanLab enabled: project={args.swanlab_project} "
            f"run={args.run} mode={args.swanlab_mode}"
        )

    return wandb.init(project=wandb_project, name=args.run, config=config)
