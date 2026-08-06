from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

from nanochat.common import DummyWandb
from nanochat import experiment_tracking


def test_tracking_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("NANOCHAT_SWANLAB_MODE", raising=False)
    parser = ArgumentParser()
    experiment_tracking.add_tracking_args(parser)
    args = parser.parse_args([])

    assert args.run == "dummy"
    assert args.swanlab_mode == "disabled"
    assert args.swanlab_project == "nanochat-lab"


def test_dummy_run_does_not_initialize_tracking(monkeypatch):
    monkeypatch.setattr(
        experiment_tracking.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected import: {name}")),
    )
    args = Namespace(run="dummy", swanlab_mode="online")

    run = experiment_tracking.init_experiment_tracking(
        args,
        {},
        master_process=True,
        wandb_project="nanochat",
        job_type="base-train",
    )

    assert isinstance(run, DummyWandb)


def test_swanlab_reuses_wandb_logging(monkeypatch):
    calls = []
    fake_swanlab = SimpleNamespace(
        init=lambda **kwargs: calls.append(("swanlab.init", kwargs)),
        sync_wandb=lambda **kwargs: calls.append(("swanlab.sync_wandb", kwargs)),
    )
    fake_wandb_run = object()
    monkeypatch.setattr(
        experiment_tracking.importlib,
        "import_module",
        lambda name: fake_swanlab if name == "swanlab" else None,
    )
    monkeypatch.setattr(
        experiment_tracking.wandb,
        "init",
        lambda **kwargs: calls.append(("wandb.init", kwargs)) or fake_wandb_run,
    )
    args = Namespace(
        run="l20-d24-base-seed42",
        swanlab_mode="offline",
        swanlab_project="nanochat-lab",
        swanlab_workspace="Qi18",
        swanlab_group="l20-d24",
        swanlab_tags="L20, bf16, d24",
    )
    config = {"depth": 24, "seed": 42}

    run = experiment_tracking.init_experiment_tracking(
        args,
        config,
        master_process=True,
        wandb_project="nanochat",
        job_type="base-train",
    )

    assert run is fake_wandb_run
    assert calls[0][0] == "swanlab.init"
    assert calls[0][1]["tags"] == ["L20", "bf16", "d24"]
    assert calls[1] == (
        "swanlab.sync_wandb",
        {"mode": "offline", "wandb_run": False},
    )
    assert calls[2] == (
        "wandb.init",
        {"project": "nanochat", "name": args.run, "config": config},
    )
