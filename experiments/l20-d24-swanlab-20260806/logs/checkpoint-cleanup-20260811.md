# Checkpoint 清理记录（2026-08-11）

## 结果

NanoChat 实验 checkpoint 已按“结果保留模式”完成清理。清理前确认无 Base、SFT、RL 或评测进程运行；共删除 218 个中间 checkpoint、probe、quarantine 和 optimizer 文件。

| 指标 | 清理前 | 清理后 |
| --- | ---: | ---: |
| Checkpoint 占用 | 224 GiB | 20 GiB |
| NanoChat cache | 239 GiB | 36 GiB |
| CPFS 已用空间 | 252 GiB（7%） | 48 GiB（2%） |
| 实际删除 | - | 218,564,384,256 bytes（203.55 GiB） |

## 保留结果

- Base final：`base_checkpoints/d24-l20-swanlab-20260806/model_011136.pt`
- SFT v1 final：`chatsft_checkpoints/d24-l20-swanlab-20260806/model_000932.pt`
- SFT v2 final：`chatsft_checkpoints/d24-l20-swanlab-20260806-sft-v2-lr02/model_000932.pt`
- RL best：`chatrl_checkpoints/d24-l20-swanlab-20260806-rl-gsm8k/model_000420.pt`
- RL final：`chatrl_checkpoints/d24-l20-swanlab-20260806-rl-gsm8k/model_000466.pt`

每个模型对应的 `meta_*.json` 同时保留。删除后重新执行 SHA256 校验，10 个模型与 metadata 文件全部通过。

## 审计文件

- 保留文件 SHA256：[`../config/checkpoint-retention-20260811.sha256`](../config/checkpoint-retention-20260811.sha256)
- 精确删除清单：[`../config/checkpoint-deletion-20260811.txt`](../config/checkpoint-deletion-20260811.txt)

数据、tokenizer、eval bundle、task data、训练与评测日志、SwanLab/W&B 记录、代码仓库和 Python 环境未删除。
