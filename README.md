# NanoChat 全流程训练实验

> Karpathy 原版项目说明保存在 [README.upstream.md](README.upstream.md)。本分支为 `experiment/l20-d24-swanlab-20260806`，实验配置、轻量日志和报告位于 [`experiments/l20-d24-swanlab-20260806/`](experiments/l20-d24-swanlab-20260806/)；模型权重与完整运行日志保留在 L20 的 CPFS。

本实验在 8×NVIDIA L20 上完成 NanoChat 的数据准备、Base 预训练、SFT、GSM8K RL 和统一评测，并使用 SwanLab 记录训练与评测过程。

## 训练方案

### 目标

1. 在 L20 上复现 NanoChat 从数据到可对话模型的完整链路。
2. 用统一协议比较 Base、SFT v1、SFT v2 和 RL checkpoint，监控专项能力与基础能力遗忘。
3. 将配置、脚本、轻量指标、SwanLab 链接、源码阅读笔记和博客草稿保存在实验分支。

### 环境与约束

| 项目 | 配置 |
| --- | --- |
| 硬件 | 8×NVIDIA L20，单卡 48 GiB |
| 分布式 | PyTorch DDP + NCCL |
| dtype | BF16，不启用 FP8 |
| 模型 | d24，24 层，hidden size 1536 |
| 数据 | ClimbMix；tokenizer 与数据缓存位于 CPFS |
| Attention | 正式训练使用 FA3，并与 PyTorch SDPA 做交叉测速 |
| 实验记录 | SwanLab；配置、摘要和轻量结果进入 GitHub |
| 安全门槛 | 8 卡 NCCL、100-step smoke、step 50 checkpoint 恢复通过后才启动 d24 |

### 阶段设计

| 阶段 | 训练目标 | 入口与配置 | 输出 |
| --- | --- | --- | --- |
| Preflight / data | 验证 GPU、NCCL、CPFS、数据和 tokenizer | [`00_preflight.sh`](experiments/l20-d24-swanlab-20260806/scripts/00_preflight.sh)、[`01_prepare_data.sh`](experiments/l20-d24-swanlab-20260806/scripts/01_prepare_data.sh) | 环境清单、数据 manifest、tokenizer 摘要 |
| Smoke / resume | d6 跑 100 step，验证 step 50 恢复 | [`02_smoke.sh`](experiments/l20-d24-swanlab-20260806/scripts/02_smoke.sh) | d6 step 50/100 checkpoint |
| Base pretrain | d24 从零训练到 step 11136 | [`03_base_train.sh`](experiments/l20-d24-swanlab-20260806/scripts/03_base_train.sh)、[`base-run.json`](experiments/l20-d24-swanlab-20260806/config/base-run.json) | `base_checkpoints/d24-l20-swanlab-20260806` |
| SFT v1 | 在 Base 上完成完整监督微调 | [`06_sft.sh`](experiments/l20-d24-swanlab-20260806/scripts/06_sft.sh) | `chatsft_checkpoints/d24-l20-swanlab-20260806` |
| SFT v2 | 降低学习率、使用 fresh optimizer，并监控 Base BPB | [`10_sft_v2.sh`](experiments/l20-d24-swanlab-20260806/scripts/10_sft_v2.sh)、[`sft-v2-run.json`](experiments/l20-d24-swanlab-20260806/config/sft-v2-run.json) | `d24-l20-swanlab-20260806-sft-v2-lr02` |
| RL | 从 SFT v2 进行 GSM8K 强化学习 | [`scripts/chat_rl.py`](scripts/chat_rl.py)、[`rl-run.json`](experiments/l20-d24-swanlab-20260806/config/rl-run.json) | `d24-l20-swanlab-20260806-rl-gsm8k` |
| Unified Eval | 统一评测 Base/SFT/RL | [`run_chat_eval_sdpa.py`](experiments/l20-d24-swanlab-20260806/scripts/run_chat_eval_sdpa.py)、[`scripts/base_eval.py`](scripts/base_eval.py) | Chat Eval、Base protocol Eval、最终报告 |

完整实验方案见 [`docs/L20_SWANLAB_EXPERIMENT_PLAN.md`](docs/L20_SWANLAB_EXPERIMENT_PLAN.md)。

## 训练过程

| 阶段 | 实际执行情况 | 关键结果 | SwanLab |
| --- | --- | --- | --- |
| Smoke | d6 step 0→50，保存并恢复到 step 100 | validation BPB `3.164625 → 1.960774 → 1.836842`；稳态约 0.9M–1.0M token/s | 离线 `tnw0wi71`、`895fo3n5` |
| d24 probe | 验证 d24 batch、显存和 checkpoint | 约 5.17 秒/step，峰值分配显存约 17.9 GiB/rank | 离线 `0spgv31a` |
| Base | step 0→11136，共 5,838,471,168 token | 957.74 分钟；最终 loss 2.35472；最低 validation BPB 0.699602 | [7malxqoi](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7malxqoi) |
| SFT v1 | Base step 11136→SFT step 932 | validation BPB `0.4394 → 0.2733`；约 105.2K token/s；评测发现基础能力遗忘 | [sjvzd160](https://swanlab.cn/@richliu0153/nanochat-lab/runs/sjvzd160) |
| SFT v2 | LR 比例 `0.8 → 0.2`，关闭旧 optimizer 动量，增加 5% warmup 和 Base BPB 监控 | step 932 的 SFT/Base validation BPB 为 `0.2820/0.7586`；显著修复 v1 遗忘 | [probe](https://swanlab.cn/@richliu0153/nanochat-lab/runs/z3ojvo4p) / [full](https://swanlab.cn/@richliu0153/nanochat-lab/runs/ceg6hcxh) |
| RL | SFT v2→GSM8K RL，共 467 step，约 2 小时 3 分钟 | step 420 pass@1/pass@8 为 `14.50%/25.25%`；step 466 最终 reward 0.1914 | [probe](https://swanlab.cn/@richliu0153/nanochat-lab/runs/6fx1yig7) / [full](https://swanlab.cn/@richliu0153/nanochat-lab/runs/6u1qttsi) |

关键过程判断：

- SFT v1 的监督损失正常收敛，但 Base BPB 和 CORE 同时退化，因此问题是分布遗忘，不是训练未收敛。
- SFT v2 通过降低学习率、清除旧优化器动量和监控 Base BPB，在对话能力与基础能力之间取得更好的平衡。
- RL 在 step 420 后收益趋于饱和，step 466 出现轻微末期退化，因此保留 step 420 作为当前 RL best。
- FA3 与 SDPA 的两轮交叉测速中，当前 SDPA 路径平均吞吐高 3.19%、峰值显存低 10.88%；结果仅代表本模型、软件版本和 L20 配置。

训练配置与报告：

- Base：[`base-final.md`](experiments/l20-d24-swanlab-20260806/logs/base-final.md)
- SFT v1：[`sft-final.md`](experiments/l20-d24-swanlab-20260806/logs/sft-final.md)、[`sft-probe.md`](experiments/l20-d24-swanlab-20260806/logs/sft-probe.md)
- SFT v2：[`sft-v2-eval.md`](experiments/l20-d24-swanlab-20260806/logs/sft-v2-eval.md)
- RL：[`rl-eval.md`](experiments/l20-d24-swanlab-20260806/logs/rl-eval.md)
- Attention：[`attention-backend-benchmark.md`](experiments/l20-d24-swanlab-20260806/logs/attention-backend-benchmark.md)
- 源码阅读与博客：[`source-notes/`](experiments/l20-d24-swanlab-20260806/source-notes/)、[`blog/drafts/`](experiments/l20-d24-swanlab-20260806/blog/drafts/)

## 评测结果

评测分为两套协议：Chat Eval 衡量指令、数学和代码能力；Base protocol Eval 衡量原始语言建模与 CORE 能力。两套协议的绝对分数不能直接互比。

### Chat Eval

统一使用 8×L20、BF16、PyTorch SDPA、temperature 0、单样本和完整测试集。

| Checkpoint | ARC-Easy | ARC-Challenge | MMLU | GSM8K | HumanEval | ChatCORE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT v1 step 932 | 52.19% | 37.20% | 34.48% | 3.18% | 14.02% | 0.1647 |
| SFT v2 step 932 | **61.41%** | **45.48%** | **36.36%** | 1.52% | **12.20%** | **0.2094** |
| RL step 420 | 59.76% | 43.77% | 36.15% | **14.71%** | 3.66% | 0.2092 |
| RL step 466 | 59.89% | 43.77% | 36.13% | 14.18% | 3.66% | 0.2084 |

### Base protocol Eval

BPB 越低越好，CORE 越高越好。

| Checkpoint | Train BPB | Validation BPB | CORE |
| --- | ---: | ---: | ---: |
| Base step 11136 | **0.714492** | **0.712537** | 0.259960 |
| SFT v1 step 932 | 0.858575 | 0.857091 | 0.227166 |
| SFT v2 step 932 | 0.767192 | 0.765257 | **0.268461** |
| RL step 420 | 0.951394 | 0.949457 | 0.258807 |
| RL step 466 | 0.957429 | 0.955496 | 0.256442 |

### 结论

- Base 正式完整 CORE 为 `0.259960`，高于原版 README 给出的 GPT-2 参考值 `0.256525`；但本实验使用 L20，不能直接参加以 8×H100 训练时间计分的 speedrun 排名。
- SFT v1 获得基础对话能力，但 Base validation BPB 上升 20.29%、CORE 下降 12.62%，存在明显遗忘。
- SFT v2 的 ChatCORE 从 `0.1647` 提升到 `0.2094`，Base CORE 从 `0.227166` 恢复到 `0.268461`，是当前综合能力最均衡的 checkpoint。
- RL step 420 将 GSM8K 从 `1.52%` 提升到 `14.71%`，但 HumanEval 降到 `3.66%`，Base validation BPB 也升至 `0.949457`；RL 是数学专项增强，并非无损的通用能力提升。

### 评测记录

| 对象 | Chat Eval | Base protocol Eval |
| --- | --- | --- |
| Base | — | [n8d4bcb1](https://swanlab.cn/@richliu0153/nanochat-lab/runs/n8d4bcb1) |
| SFT v1 | [i239hpv4](https://swanlab.cn/@richliu0153/nanochat-lab/runs/i239hpv4) | [ln0ha6dv](https://swanlab.cn/@richliu0153/nanochat-lab/runs/ln0ha6dv) |
| SFT v2 | [0p6h93ke](https://swanlab.cn/@richliu0153/nanochat-lab/runs/0p6h93ke) | [9ut6s28v](https://swanlab.cn/@richliu0153/nanochat-lab/runs/9ut6s28v) |
| RL step 420/466 | [7v4ta558](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7v4ta558) | [7v4ta558](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7v4ta558) |

轻量 CSV、原始日志摘要和 SHA256 均保存在 [`experiments/l20-d24-swanlab-20260806/logs/`](experiments/l20-d24-swanlab-20260806/logs/)。
