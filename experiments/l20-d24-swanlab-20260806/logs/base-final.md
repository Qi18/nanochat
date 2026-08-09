# d24 Base Pretrain 最终记录

## 结论

8×NVIDIA L20、BF16、Flash Attention 3、d24 的 Base Pretrain 已完成。11,136 个优化 step 和 5,838,471,168 个训练 token 全部执行成功，无 NaN、OOM 或 NCCL 错误；最终 checkpoint 完整保存，SwanLab 离线 run 已同步到云端。

## 配置

- 模型参数量：1,384,122,122
- scaling parameters：729,810,624
- depth / width / heads：24 / 1536 / 12
- sequence length：2048
- device batch：2
- total batch：524,288 token
- gradient accumulation：16
- dtype：bfloat16
- attention backend：Flash Attention 3
- target parameter/data ratio：8
- 训练 step：11,136

## 最终结果

| 指标 | 结果 |
| --- | ---: |
| 总训练时间 | 957.74 分钟（15 小时 57 分 44.67 秒） |
| 最终训练 loss | 2.35472 |
| 最低 validation bpb | 0.699602 |
| 最终 CORE metric | 0.27082 |
| 最终 step 时间 | 5.16538 秒 |
| 最终吞吐 | 约 101,500 token/s |
| 峰值 allocated memory | 17,890.36 MiB/rank |
| 总训练 FLOPs | 3.085534832103103e+19 |
| GPU 时 | 约 127.7 GPU-hours |

NanoChat 在 L20 上无法识别理论峰值 FLOPS，因此日志中的 `bf16_mfu=0` 不可用于性能判断；本实验以 step wall time、token/s 和显存作为速度证据。

## 产物

- checkpoint 目录：`/data/cache/nanochat/base_checkpoints/d24-l20-swanlab-20260806`
- model：`model_011136.pt`
- metadata：`meta_011136.json`
- optimizer：`optim_011136_rank0.pt` 至 `optim_011136_rank7.pt`
- 原始日志：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/base-train.log`

## SwanLab

- project：<https://swanlab.cn/@richliu0153/nanochat-lab>
- run：<https://swanlab.cn/@richliu0153/nanochat-lab/runs/7malxqoi>
- 同步记录数：335,814
- 同步完成时间：2026-08-07 15:34:19（Asia/Shanghai）

## 完整性判断

- 11,136/11,136 step 完成。
- 最终 validation、CORE 和采样完成。
- model、metadata 和 8 个 optimizer shard 完成落盘。
- 日志未出现 NaN、OOM 或 NCCL 错误。
- 下一步先执行同配置 FA3/SDPA 训练速度对比，再进入 Base Eval。
