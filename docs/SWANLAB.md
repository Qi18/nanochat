# SwanLab experiment tracking

Nanochat keeps its existing `wandb.log` calls and can optionally mirror them
into one SwanLab project. Tracking is disabled by default, so upstream commands
continue to behave exactly as before.

## Install

```bash
uv sync --extra gpu --extra tracking
# CPU/MPS: uv sync --extra cpu --extra tracking
swanlab login --local
```

## L20 run setup

Use one project and group the base, SFT, and RL legs together:

```bash
export NANOCHAT_SWANLAB_MODE=online
export NANOCHAT_SWANLAB_PROJECT=nanochat-lab
export NANOCHAT_SWANLAB_WORKSPACE=Qi18
export NANOCHAT_SWANLAB_GROUP=l20-d24-2026-08
export NANOCHAT_SWANLAB_TAGS=L20,bf16,d24,8gpu
```

Give every leg a unique `--run` value; `--run=dummy` intentionally disables
both W&B and SwanLab.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train -- \
  --run=l20-d24-base-seed42 \
  --depth=24 \
  --window-pattern=L

torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_sft -- \
  --run=l20-d24-sft-seed42

torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_rl -- \
  --run=l20-d24-rl-gsm8k-seed42
```

Only DDP rank 0 initializes SwanLab. `wandb_run=False` preserves nanochat's W&B
logging API without uploading the experiment to W&B.

For unstable networks, use `NANOCHAT_SWANLAB_MODE=offline`, preserve `swanlog/`
outside Git, and upload it later with `swanlab sync <logdir>`.

Commit commands, configs, metrics, reports, and checkpoint SHA256 manifests to
the Fork. Keep datasets, `.pt` checkpoints, and large raw logs in `/data/cache`
or object storage.
