# RL 训练与评测报告

## 1. 实验设置

本轮从 SFT v2 的 `d24-l20-swanlab-20260806-sft-v2-lr02` step 932 继续做 GSM8K RL，输出 `d24-l20-swanlab-20260806-rl-gsm8k`。训练使用 8×L20、BF16、FA3，1 个 epoch，共 467 step；每步 16 个采样、最多生成 256 token，temperature=1、top-k=50，评估集 400 条，每 60 step 评估并保存 checkpoint。完整训练时间约 2 小时 3 分钟。

训练内评估（400 条）如下：

| step | pass@1 | pass@8 |
| ---: | ---: | ---: |
| 0 | 1.25% | 9.00% |
| 60 | 5.00% | 16.50% |
| 120 | 13.50% | 27.75% |
| 180 | 14.25% | 26.50% |
| 240 | 10.00% | 22.00% |
| 300 | 12.00% | 22.25% |
| 360 | 14.00% | 24.25% |
| 420 | 14.50% | 25.25% |

step 466 是最终保存点，但未触发 60-step 周期评估；最终训练 batch reward 为 0.1914。因此选择 step 420 作为主要 best checkpoint，step 466 作为 final checkpoint。

训练过程 SwanLab：[6u1qttsi](https://swanlab.cn/@richliu0153/nanochat-lab/runs/6u1qttsi)。

## 2. 独立 Chat Eval

统一使用 PyTorch SDPA、temperature=0、num_samples=1、max_new_tokens=512；完整测试集结果：

| checkpoint | ARC-Easy | ARC-Challenge | MMLU | GSM8K | HumanEval | ChatCORE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT v2 step 932 | 61.41% | 45.48% | 36.36% | 1.52% | 12.20% | 0.2094 |
| RL step 420 | 59.76% | 43.77% | 36.15% | 14.71% | 3.66% | 0.2092 |
| RL step 466 | 59.89% | 43.77% | 36.13% | 14.18% | 3.66% | 0.2084 |

RL 主要提升了 GSM8K（相对 SFT v2 +13.19 个百分点），但 HumanEval 下降 8.54 个百分点；综合 ChatCORE 基本持平，step 466 比 step 420 低 0.0008，存在轻微末期退化。

原始日志和轻量 CSV：[rl-chat-eval-step420.log](rl-chat-eval-step420.log)、[rl-chat-eval-step466.log](rl-chat-eval-step466.log)、[rl-chat-eval.csv](rl-chat-eval.csv)。

## 3. Base protocol Eval

| checkpoint | train BPB | val BPB | CORE |
| --- | ---: | ---: | ---: |
| Base step 11136 | 0.714492 | 0.712537 | 0.259960 |
| SFT v2 step 932 | 0.767192 | 0.765257 | 0.268461 |
| RL step 420 | 0.951394 | 0.949457 | 0.258807 |
| RL step 466 | 0.957429 | 0.955496 | 0.256442 |

相对 SFT v2，RL step 420 的 Base CORE 下降 0.009654（-3.60%），step 466 下降 0.012019（-4.48%）；相对原始 Base，step 420 低 0.001153（-0.44%），step 466 低 0.003518（-1.35%）。Base val BPB 也持续上升，说明 GSM8K reward 优化带来了基础语言建模能力损失。

原始结果：[rl-base-protocol-step420.csv](rl-base-protocol-step420.csv)、[rl-base-protocol-step466.csv](rl-base-protocol-step466.csv)、[rl-base-protocol.csv](rl-base-protocol.csv)。

## 4. 结论与后续

本轮 RL 已验证“对 GSM8K 有效，但不是无损通用能力提升”：step 420 是当前 Pareto 更好的选择，适合保留为 RL best；step 466 不建议作为默认发布点。若继续优化，应增加 KL 约束或参考模型、降低 RL 学习率/训练步数，并对 GSM8K、HumanEval、ChatCORE 和 Base CORE 设联合早停条件；在当前目标是 NanoChat leaderboard 时不应把 RL 结果作为 pretraining entry。

完整评测指标已上传 SwanLab：[7v4ta558](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7v4ta558)。

## 5. Provenance

原始日志位于 `/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/`；提交到实验目录的评测日志 SHA256：

- `rl-chat-eval-step420.log`: `d05eb829d16ce901a9c6e338a8b0cd11c3e87be595b599687dadfcec0fb257d7`
- `rl-chat-eval-step466.log`: `669f321a90fa7ad2de8b909b0b5fc08b983963721af3c950b4b2c11850d276b6`
- `rl-base-eval-step420.log`: `0700f260c8be42d4f26c8700a712c9658fe68d4f7668b988d1958fe2e92eff7e`
- `rl-base-eval-step466.log`: `eba2abc2a243bdb2d96d3c1266c7092e058d50410ee2cc10a33b4285cb62fe0d`

评测时仓库 HEAD 为 `8c835667c3f8bdf5c5abbd9cb46ed2c41eb4edd3`；工作树保留用户既有未提交改动，未执行 commit 或 push。
