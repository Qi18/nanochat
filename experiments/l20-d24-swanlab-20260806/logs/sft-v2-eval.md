# SFT v2 评测记录

评测对象：`d24-l20-swanlab-20260806-sft-v2-lr02`，checkpoint step 932，8×L20，BF16，PyTorch SDPA。完整 Chat Eval 和 Base protocol 均在 L20 上独立运行完成。

## 结果

### 完整 Chat Eval

| 任务 | 正确/总数 | Accuracy |
|---|---:|---:|
| ARC-Easy | 1459/2376 | 61.41% |
| ARC-Challenge | 533/1172 | 45.48% |
| MMLU | 5106/14042 | 36.36% |
| GSM8K | 20/1319 | 1.52% |
| HumanEval | 20/164 | 12.20% |
| ChatCORE | — | 0.2094 |

完整原始日志：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/sft-v2-full-chat-eval.log`。

### Base 同协议 BPB/CORE

| 指标 | Base step11136 | SFT v1 step932 | SFT v2 step932 |
|---|---:|---:|---:|
| train BPB | 0.714492 | — | 0.767192 |
| val BPB | 0.712537 | 0.857091 | 0.765257 |
| CORE | 0.259960 | 0.227166 | 0.268461 |

v2 相对 Base：val BPB +0.052720（+7.40%），CORE +0.008501（+3.27%）。v2 相对 SFT v1：val BPB -0.091834（-10.71%），CORE +0.041295（+18.17%）。BPB 越低越好，CORE 越高越好。

## 与 v1 的 Chat Eval 对比

| 任务 | SFT v1 | SFT v2 | 变化（百分点） |
|---|---:|---:|---:|
| ARC-Easy | 52.19% | 61.41% | +9.22 |
| ARC-Challenge | 37.20% | 45.48% | +8.28 |
| MMLU | 34.48% | 36.36% | +1.88 |
| GSM8K | 3.18% | 1.52% | -1.66 |
| HumanEval | 14.02% | 12.20% | -1.82 |
| ChatCORE | 0.164724 | 0.2094 | +0.044676 |

结论：v2 已明显修复 v1 的 Base 能力退化，且 ARC/MMLU/ChatCORE 上升；但 GSM8K、HumanEval 仍弱，不能称为通用 leaderboard 高水平模型。当前更准确的定位是“完成可复现实验闭环、具备基础问答能力的小模型”，不是与 7B/14B 级公开榜单直接同档。GSM8K 和 HumanEval 的结果还应结合更长训练、推理模板和多次采样复核。

## SwanLab

- v2 训练（离线记录已同步）：https://swanlab.cn/@richliu0153/nanochat-lab/runs/ceg6hcxh
- v2 完整 Chat Eval：https://swanlab.cn/@richliu0153/nanochat-lab/runs/0p6h93ke
- v2 Base protocol：https://swanlab.cn/@richliu0153/nanochat-lab/runs/9ut6s28v

原始日志哈希：

- Chat Eval：`d0a4b56d6bb7ceb968c8ee241ecb3c7849d2b5af43bdb9645c0d141319e6c0ae`
- Base protocol：`0f45db03d1feb4b3f55d9e5c65b1f1b76bdf8d071709f8a32aeb40429dbe3298`
- Base protocol CSV：`445edacf7b0abf75f73b49e669c5646c91f0ecce7a7216e179d302d6bb984526`
