# d24 SFT 最终记录

## 结论

8×NVIDIA L20 的 d24 SFT 已完成 932 step，训练过程无 NaN、OOM 或 NCCL 错误。validation BPB 从 0.4394 降至 0.2733，最终 ChatCORE 为 0.17447。checkpoint、metadata 与 8 份 optimizer shard 均已保存。

SwanLab：[l20-d24-swanlab-20260806-sft](https://swanlab.cn/@richliu0153/nanochat-lab/runs/sjvzd160)，已上传 28,622 条记录。

## 配置

- 时间：2026-08-07 09:53:18 UTC 至 11:15:42 UTC；
- Base checkpoint：`d24-l20-swanlab-20260806/model_011136.pt`；
- 硬件：8×NVIDIA L20；
- dtype：BF16；
- attention：PyTorch SDPA；
- sequence length：2,048；
- device batch：2；
- total batch：524,288 tokens；
- gradient accumulation：16；
- 数据混合：789,759 行，MMLU x3、GSM8K x4；
- validation：每 100 step，4,194,304 tokens；
- ChatCORE：每 200 step，分类任务最多 500 条、生成任务最多 24 条。

## 训练结果

- optimizer step：932；
- 训练计时：4,590.39 秒（76.51 分钟，不含阶段性评估）；
- 最终 train loss：0.928801；
- 最终稳态：4,981.22 ms/step，105,252 token/s；
- 峰值显存：15,943.26 MiB/rank；
- 初始 / 最终 validation BPB：0.4394 / 0.2733；
- 最低 validation BPB：0.2733。

validation BPB 轨迹：

| Step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 932 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPB | 0.4394 | 0.3383 | 0.3244 | 0.3202 | 0.3153 | 0.3092 | 0.2993 | 0.2896 | 0.2805 | 0.2741 | 0.2733 |

## 最终 ChatCORE

| 指标 | 结果 |
| --- | ---: |
| ARC-Easy | 53.60% |
| ARC-Challenge | 37.40% |
| MMLU | 33.80% |
| GSM8K | 4.17% |
| HumanEval | 16.67% |
| ChatCORE categorical | 0.22133 |
| ChatCORE | 0.17447 |

这里的 ChatCORE 是训练内评估：分类任务最多 500 条、生成任务最多 24 条，不能与后续完整统一评测混为一谈。

## 产物与校验

checkpoint：`/data/cache/nanochat/chatsft_checkpoints/d24-l20-swanlab-20260806`

- `model_000932.pt`；
- `meta_000932.json`；
- `optim_000932_rank0.pt` 至 `optim_000932_rank7.pt`。

原始产物保存在 CPFS，不进入 Git：

- log：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/sft.log`；
- SwanLab：`/data/cache/swanlab/l20-d24-swanlab-20260806/run-20260807_095320-v311fsio`；
- log SHA256：`6638d980d3601164def637962104519cbeeb1642c6cbe8b4cd8b205198adbb0e`；
- metadata SHA256：`e4211bbe63de829babbf05ecef4f0db0fd791e5bcc0a7e3b6f3cc65b6bc7a3eb`；
- SwanLab sync log SHA256：`67624fb69ae83f3e0e452fe4d22b8f2b2ddbd60379995baa5e4f89c6fa0187b7`。

结构化结果见 `../config/sft-results.json`，启动前探针与修复见 `sft-probe.md`。
