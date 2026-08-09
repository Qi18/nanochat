# SFT v2 改进对照启动记录

## 目标

v1 SFT 的全量同协议评估显示：Base val BPB 增加 20.29%，CORE 相对下降 12.62%。v2 优先降低更新强度，并在训练过程中直接监控 Base validation BPB，避免只根据 SFT validation 选择最后一步。

## 与 v1 的差异

| 配置 | v1 | v2 |
| --- | ---: | ---: |
| `init_lr_frac` | 0.8 | 0.2 |
| `load_optimizer` | 1 | 0 |
| `warmup_ratio` | 0 | 0.05 |
| `warmdown_ratio` | 0.5 | 0.3 |
| Base validation BPB | 训练后单独评估 | 每 100 step 监控 1,048,576 tokens |
| checkpoint | 最终一步 | 每 200 step 模型 + 最终完整 optimizer |
| 输出目录 | `d24-l20-swanlab-20260806` | `d24-l20-swanlab-20260806-sft-v2-lr02` |

旧 SFT checkpoint 不会被覆盖。中间 checkpoint 只保存模型，最终 checkpoint 保存模型、metadata 和 8 份 optimizer shard。

## 3-step 探针

- 状态：通过；
- SFT validation BPB：0.4225；
- Base validation BPB proxy：0.7163 → 0.7201；
- 峰值显存：15,941.77 MiB/rank；
- checkpoint：独立目录中 model、metadata、8 份 optimizer shard 完整；
- SwanLab：[z3ojvo4p](https://swanlab.cn/@richliu0153/nanochat-lab/runs/z3ojvo4p)。

## 完整 v2 当前状态

- 启动时间：2026-08-07 14:02:24 UTC；
- SwanLab：offline，local run `sgy2x6zx`；
- step 1 LR multiplier：0.02；
- step 50 LR multiplier：1.00，5% warmup 正常完成；
- 稳态吞吐：约 105.5K token/s；
- 当前已验证至 step 100，无 NaN、OOM 或 NCCL 错误。

| Step | SFT validation BPB | Base validation BPB proxy |
| ---: | ---: | ---: |
| 0 | 0.4394 | 0.7090 |
| 100 | 0.3524 | 0.7309 |

v1 在 step 100 的 SFT validation BPB 为 0.3383。v2 对话目标收敛更慢，但 Base proxy 当前只增加约 3.1%；是否优于 v1 必须结合后续 step 200/400/600/800 checkpoint 的 ChatCORE 与同协议 CORE 评估判断。

## 产物

- 配置：`../config/sft-v2.env`；
- 结构化状态：`../config/sft-v2-run.json`；
- 探针脚本：`../scripts/09_sft_v2_probe.sh`；
- 正式脚本：`../scripts/10_sft_v2.sh`；
- 原始日志：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/sft-v2.log`；
- 正式 checkpoint：`/data/cache/nanochat/chatsft_checkpoints/d24-l20-swanlab-20260806-sft-v2-lr02`。
