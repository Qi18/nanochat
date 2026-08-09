"""End-to-end D24 training-step benchmark for FA3 versus PyTorch SDPA.

This intentionally follows scripts/base_train: real ClimbMix batches, torch.compile,
16-way gradient accumulation, backward, distributed MuonAdamW communication, and
optimizer updates. It does not evaluate, track, or save checkpoints.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist

import nanochat.flash_attention as fa_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("fa3", "sdpa"), required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main():
    args = parse_args()

    # Resolve the backend before GPT captures the unified flash_attn interface.
    fa_module._override_impl = args.backend
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()
    assert fa_module.USE_FA3 == (args.backend == "fa3")

    from nanochat.common import COMPUTE_DTYPE, compute_cleanup, compute_init, print0
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer

    ddp, rank, local_rank, world_size, device = compute_init("cuda")
    assert ddp and world_size == 8, (ddp, world_size)
    assert COMPUTE_DTYPE == torch.bfloat16, COMPUTE_DTYPE
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    depth = 24
    aspect_ratio = 64
    head_dim = 128
    sequence_len = 2048
    device_batch_size = 2
    total_batch_tokens = 524288
    width = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    num_heads = width // head_dim
    tokens_per_microbatch = device_batch_size * sequence_len * world_size
    grad_accum_steps = total_batch_tokens // tokens_per_microbatch
    assert grad_accum_steps == 16

    tokenizer = get_tokenizer()
    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=width,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    # Match Base Pretrain optimizer hyperparameters. The d24 weight-decay scaling
    # is derived from the d12 reference with the same target data:param ratio.
    with torch.device("meta"):
        d12_width = ((12 * aspect_ratio + head_dim - 1) // head_dim) * head_dim
        d12_model = GPT(GPTConfig(
            sequence_len=sequence_len,
            vocab_size=tokenizer.get_vocab_size(),
            n_layer=12,
            n_head=d12_width // head_dim,
            n_kv_head=d12_width // head_dim,
            n_embd=d12_width,
            window_pattern="L",
        ))
    scaling_params = model.num_scaling_params()
    d12_scaling_params = d12_model.num_scaling_params()
    d24_scale = scaling_params["transformer_matrices"] + scaling_params["lm_head"]
    d12_scale = d12_scaling_params["transformer_matrices"] + d12_scaling_params["lm_head"]
    weight_decay = 0.28 * (d12_scale / d24_scale)
    del d12_model

    optimizer = model.setup_optimizer(
        unembedding_lr=0.008,
        embedding_lr=0.3,
        scalar_lr=0.5,
        matrix_lr=0.02,
        weight_decay=weight_decay,
    )
    model = torch.compile(model, dynamic=False)
    model.train()

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        device_batch_size,
        sequence_len,
        split="train",
        device=device,
        resume_state_dict=None,
    )
    x, y, _ = next(train_loader)

    print0(
        f"benchmark backend={args.backend} warmup={args.warmup_steps} "
        f"measure={args.measure_steps} grad_accum={grad_accum_steps}"
    )
    durations = []
    losses = []
    total_steps = args.warmup_steps + args.measure_steps
    for step in range(total_steps):
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, _ = next(train_loader)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        train_loss_value = train_loss.item()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        elapsed_max = elapsed_tensor.item()

        phase = "warmup" if step < args.warmup_steps else "measure"
        print0(
            f"{phase} step={step:02d} dt={elapsed_max:.5f}s "
            f"tok/s={total_batch_tokens / elapsed_max:.0f} loss={train_loss_value:.6f}"
        )
        if step == args.warmup_steps - 1:
            torch.cuda.reset_peak_memory_stats()
        if step >= args.warmup_steps:
            durations.append(elapsed_max)
            losses.append(train_loss_value)

    peak_memory = torch.tensor(
        torch.cuda.max_memory_allocated() / (1024 ** 2),
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

    if rank == 0:
        mean_seconds = statistics.fmean(durations)
        result = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "attention_backend": "flash_attention_3" if args.backend == "fa3" else "pytorch_sdpa",
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(local_rank),
            "dtype": str(COMPUTE_DTYPE).removeprefix("torch."),
            "model": {
                "depth": depth,
                "width": width,
                "heads": num_heads,
                "sequence_length": sequence_len,
                "parameters": scaling_params["total"],
                "window_pattern": "L",
            },
            "batch": {
                "device_batch_size": device_batch_size,
                "total_batch_tokens": total_batch_tokens,
                "gradient_accumulation_steps": grad_accum_steps,
            },
            "protocol": {
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "seed": args.seed,
                "includes": ["real_data", "forward", "backward", "distributed_optimizer", "optimizer_step"],
                "excludes": ["evaluation", "tracking", "checkpoint"],
            },
            "step_seconds": durations,
            "losses": losses,
            "mean_step_seconds": mean_seconds,
            "median_step_seconds": statistics.median(durations),
            "p95_step_seconds": percentile(durations, 0.95),
            "mean_tokens_per_second": total_batch_tokens / mean_seconds,
            "peak_allocated_memory_mib": peak_memory.item(),
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    compute_cleanup()


if __name__ == "__main__":
    main()
