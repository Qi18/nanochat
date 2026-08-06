# Smoke Attempt 2

## 结论

通过。8×NVIDIA L20、BF16、d6、序列长度 2048 的 100-step smoke 完整执行，并从 step 50 checkpoint 恢复到 step 100。

## 运行信息

- 启动 commit：`ec3b4408e7b3d41cc18f6360fab54087ca380410`
- Torch：2.6.0+cu124
- dtype：bfloat16
- Attention：PyTorch SDPA fallback（FA3 unavailable）
- device batch：2
- total batch：65536 tokens
- gradient accumulation：2
- 模型参数量：73,531,538
- 训练数据：171 个完整 shard

## 指标

| step | train loss | validation bpb |
| ---: | ---: | ---: |
| 0 | 10.397406 | 3.164625 |
| 25 | 8.147515 | 2.224372 |
| 50 | 6.575670（恢复后首步） | 1.960774 |
| 75 | 6.257261 | 1.871712 |
| 100 | - | 1.836842 |

- 编译后的稳态吞吐：约 0.9M–1.0M token/s。
- 首次编译 step：25,590 ms。
- 恢复进程首 step：6,361 ms。
- 峰值显存：约 1.1 GiB/rank。
- smoke 输出中未出现 NaN、Inf 或 CUDA OOM。

## Checkpoint 恢复证据

step 50 与 step 100 各包含：

- 1 个 `model_<step>.pt`
- 1 个 `meta_<step>.json`
- 8 个 `optim_<step>_rankN.pt`

dataloader state：

- step 50：`epoch=1,pq_idx=0,rg_idx=8`
- step 100：`epoch=1,pq_idx=0,rg_idx=24`

## SwanLab

- 首段 run ID：`tnw0wi71`
- 恢复段 run ID：`895fo3n5`
- 当前模式：offline
- 原始 run 保存在 CPFS，尚未同步云端。
- smoke 发现 `train/epoch` 复合字符串不被 SwanLab 0.9.2 接受；已拆为三个整数指标，单元测试与真实离线 tracking smoke 均通过。

## 产物哈希

- supervisor log：`e0ed1c60949f6b7ecd91a97b476ea3b41cd80815b49832cc15efcdedc9989944`
- model step 50：`82540becc1bd890b1973c195c5482f7ef73257d983c020f64eef8b671b2648d2`
- model step 100：`cb5f0e53dad8db98410b923fd8d8970a9827afe448e0262fa549df60607f1b36`
- SwanLab 首段：`83ef31b5b130adba0cbea975a700514fbf1fc0b940f500dcb2d1db6af46ec2e8`
- SwanLab 恢复段：`5ba085993283d27e6532e7cbe4be187ebb16f4e87a98f189d3b146b070f97d2c`

完整原始日志、checkpoint 与 SwanLab run 只保存在 CPFS；GitHub 仅保存此摘要和哈希。
