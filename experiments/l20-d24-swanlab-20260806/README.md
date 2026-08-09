# l20-d24-swanlab-20260806

NanoChat 在 8×NVIDIA L20 上的当前代码基线复现实验，使用 SwanLab 记录 Base、SFT、RL 和评测过程，并同步产出源码阅读笔记与博客草稿。

## 当前状态

- [x] 创建 L20 独立实验分支
- [x] ACK API、真实 SSH、8×L20、CPFS 和共享内存检查
- [x] PyTorch CUDA Tensor 与 8 卡 NCCL all-reduce
- [x] SwanLab/W&B/rustbpe/kernels 隔离环境
- [x] Hugging Face 直连与镜像验证
- [x] 固化 provenance
- [x] 数据与 tokenizer
- [x] 100-step checkpoint 恢复冒烟
- [x] d24 Base（11,136 step）
- [x] FA3 / PyTorch SDPA 训练速度对比
- [x] Base Eval
- [x] SFT
- [x] SFT 独立全量评估与 Base 同协议对比
- [x] SFT v2 低学习率 / fresh optimizer / Base retention 对照
- [x] RL（GSM8K，467 step，SFT v2 → RL）
- [x] 统一 Chat Eval、Base protocol Eval 与最终报告

## 最近结果

- 8 卡 BF16 d6 smoke 在 step 50 保存后，从完整 checkpoint 恢复至 step 100。
- validation bpb：3.164625（step 0）→ 1.960774（step 50）→ 1.836842（step 100）。
- 稳态吞吐约 0.9M–1.0M token/s；峰值显存约 1.1 GiB/rank。
- step 50 与 step 100 均有模型、metadata 和 8 份 optimizer state。
- 两个 smoke 阶段的 SwanLab 离线 run 已落盘：tnw0wi71、895fo3n5；未计入已同步云端训练记录。
- 对齐 `kernels==0.11.7` 并修复 Torch 2.6 graph-break/tuple 兼容后，FA3 GPU 回归 20 项通过。
- d24 正式 batch 探针稳态约 5.17 秒/step、峰值分配显存 17.9 GiB/rank；正式 Base 预计约 17–19 小时。
- d24 Base 已完成 11,136 step / 5,838,471,168 token：总训练时间 957.74 分钟，最终训练 loss 2.35472，最低 validation bpb 0.699602，CORE 0.27082。
- 最终 checkpoint、metadata 与 8 份 optimizer shard 均已保存；SwanLab run 已同步到云端。
- 两轮交叉测速中，PyTorch SDPA 相比当前 FA3 路径平均 step 时间降低 3.09%、吞吐提高 3.19%、峰值显存降低 10.88%。
- 独立 Base Eval 已完成：train/val BPB 0.714492/0.712537，完整 CORE 0.259960；结果已上传 SwanLab。
- d24 SFT 已完成 932 step：validation BPB 0.4394 → 0.2733，最终 ChatCORE 0.17447，稳态约 105.2K token/s，峰值显存 15,943 MiB/rank。
- SFT checkpoint、metadata 和 8 份 optimizer shard 已保存；SwanLab 28,622 条记录已同步到云端。
- SFT 独立全量 Chat Eval：ARC-Easy 52.19%、ARC-Challenge 37.20%、MMLU 34.48%、GSM8K 3.18%、HumanEval 14.02%，ChatCORE 0.1647。
- Base 同协议评估显示基础能力遗忘：val BPB 0.712537 → 0.857091，CORE 0.259960 → 0.227166；22 项中 6 升、1 持平、15 降。
- SFT v2 已完成独立对照：LR 比例 0.8 → 0.2、关闭 Base optimizer 动量、增加 5% warmup 和 Base BPB 监控；step 932 SFT/Base validation BPB 为 0.2820/0.7586。完整 Chat Eval 为 ARC-Easy 61.41%、ARC-Challenge 45.48%、MMLU 36.36%、GSM8K 1.52%、HumanEval 12.20%，ChatCORE 0.2094；Base 同协议 val BPB/CORE 为 0.765257/0.268461。详见 [logs/sft-v2-eval.md](logs/sft-v2-eval.md)。
- RL 已从 SFT v2 step 932 训练 467 step，8×L20 BF16，输出 `d24-l20-swanlab-20260806-rl-gsm8k`；训练内 step 420 的 GSM8K pass@1/pass@8 为 14.50%/25.25%，最终 step 466 reward 为 0.1914。训练 SwanLab：[6u1qttsi](https://swanlab.cn/@richliu0153/nanochat-lab/runs/6u1qttsi)。
- RL 完整评测：step 420 ChatCORE 0.2092、Base CORE 0.2588；step 466 ChatCORE 0.2084、Base CORE 0.2564。GSM8K 从 SFT v2 的 1.52% 提升至 14.71%（step 420），但 HumanEval 从 12.20% 降至 3.66%；step 466 相比 step 420 略有末期退化。评测 SwanLab：[7v4ta558](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7v4ta558)，详细结论见 [logs/rl-eval.md](logs/rl-eval.md)。
- SFT 评估与对比见 [logs/sft-eval.md](logs/sft-eval.md)，训练记录见 [logs/sft-final.md](logs/sft-final.md)，探针与修复见 [logs/sft-probe.md](logs/sft-probe.md)，attention 对比见 [logs/attention-backend-benchmark.md](logs/attention-backend-benchmark.md)。

## 完整训练流程

本实验按“环境与数据 → 冒烟 → d24 probe → Base 预训练 → SFT → RL → 统一评测”执行；正式训练和评测均在 L20 分支 `experiment/l20-d24-swanlab-20260806` 上完成。

| 阶段 | 入口/配置 | 输入 → 输出 | 记录与产物 |
| --- | --- | --- | --- |
| 1. Preflight / data | `scripts/00_preflight.sh`、`01_prepare_data.sh` | L20、8 卡 NCCL、CPFS、ClimbMix、tokenizer | `config/environment.env`、`config/data_manifest.json`、`source-notes/` |
| 2. Smoke / resume | `scripts/02_smoke.sh` | d6 step 0 → 100，step 50 保存并恢复 | `config/smoke-results.json`；SwanLab 离线 run `tnw0wi71`、`895fo3n5` |
| 3. d24 probe | d24 batch / checkpoint probe | 24 层、1536 hidden、BF16、FA3 | `config/d24-probe-results.json`；离线 run `0spgv31a` |
| 4. Base pretrain | `scripts/03_base_train.sh` | Base step 0 → 11,136，约 5.84B tokens | checkpoint `base_checkpoints/d24-l20-swanlab-20260806`；SwanLab [7malxqoi](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7malxqoi) |
| 5. Attention benchmark | `scripts/attention_backend_benchmark.py` | FA3 vs PyTorch SDPA | `logs/attention-backend-benchmark.md` |
| 6. SFT v1 | `scripts/05_sft_probe.sh`、`06_sft.sh` | Base step 11136 → SFT step 932 | checkpoint `chatsft_checkpoints/d24-l20-swanlab-20260806`；SwanLab [sjvzd160](https://swanlab.cn/@richliu0153/nanochat-lab/runs/sjvzd160) |
| 7. SFT v2 | `scripts/09_sft_v2_probe.sh`、`10_sft_v2.sh` | SFT v1 step 932 → `d24-l20-swanlab-20260806-sft-v2-lr02` step 932 | probe [z3ojvo4p](https://swanlab.cn/@richliu0153/nanochat-lab/runs/z3ojvo4p)，full [ceg6hcxh](https://swanlab.cn/@richliu0153/nanochat-lab/runs/ceg6hcxh) |
| 8. RL / GSM8K | `scripts/chat_rl.py`，配置见 `config/rl-run.json` | SFT v2 step 932 → `d24-l20-swanlab-20260806-rl-gsm8k` step 466 | probe [6fx1yig7](https://swanlab.cn/@richliu0153/nanochat-lab/runs/6fx1yig7)，full [6u1qttsi](https://swanlab.cn/@richliu0153/nanochat-lab/runs/6u1qttsi) |
| 9. Unified Eval | `scripts/run_chat_eval_sdpa.py`、`scripts/base_eval.py` | Base / SFT / RL 的 Chat Eval 与 Base protocol Eval | `logs/*-chat-eval.csv`、`logs/*-base-protocol.csv`、`logs/rl-eval.md` |
评测 SwanLab：Base [n8d4bcb1](https://swanlab.cn/@richliu0153/nanochat-lab/runs/n8d4bcb1)；SFT v1 [i239hpv4](https://swanlab.cn/@richliu0153/nanochat-lab/runs/i239hpv4) / [ln0ha6dv](https://swanlab.cn/@richliu0153/nanochat-lab/runs/ln0ha6dv)；SFT v2 [0p6h93ke](https://swanlab.cn/@richliu0153/nanochat-lab/runs/0p6h93ke) / [9ut6s28v](https://swanlab.cn/@richliu0153/nanochat-lab/runs/9ut6s28v)；RL [7v4ta558](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7v4ta558)。

### 执行顺序

1. 完成环境、网络、GPU、NCCL、数据和 tokenizer 检查。
2. 完成 d6 100-step 冒烟，并验证 step 50 checkpoint 可恢复。
3. 完成 d24 probe 和 FA3/SDPA 对比，确认正式训练配置。
4. 训练 Base 到 step 11136，保存模型、metadata 和 optimizer shards。
5. 用 Base checkpoint 进行 SFT v1；发现基础能力遗忘后，降低学习率并关闭旧 optimizer 动量，完成 SFT v2 对照。
6. 从 SFT v2 step 932 进入 GSM8K RL；每 60 step 做 pass@k 评估并保存 checkpoint。
7. 对 SFT v2、RL step 420（best）和 RL step 466（final）执行统一 Chat Eval 与 Base protocol Eval。
8. 将训练、评测指标和报告写入本目录；SwanLab 原始目录和大 checkpoint 继续保留在 CPFS。

### 关键产物

- 训练配置：`config/base-run.json`、`config/sft-v2-run.json`、`config/rl-run.json`。
- 训练摘要：`logs/base-final.md`、`logs/sft-final.md`、`logs/sft-v2-eval.md`、`logs/rl-eval.md`。
- 统一指标：`logs/rl-chat-eval.csv`、`logs/rl-base-protocol.csv` 及对应原始日志。
- 完整方案与源码阅读：`../../docs/L20_SWANLAB_EXPERIMENT_PLAN.md`、`source-notes/`、`blog/drafts/`。

## 不可变约束

- Git 操作只在 L20 执行。
- 正式训练使用 BF16，不启用 FP8。
- 未通过 8 卡 100-step 冒烟和 step 50 恢复验证，不启动 d24。
- 数据、完整日志、SwanLab 原始目录和 checkpoint 保存在 CPFS，不进入 Git。
- GitHub 保存配置、轻量指标、日志摘要、图表、哈希和报告。

完整方案见 [../../docs/L20_SWANLAB_EXPERIMENT_PLAN.md](../../docs/L20_SWANLAB_EXPERIMENT_PLAN.md)。
