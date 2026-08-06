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
- [ ] d24 Base
- [ ] Base Eval
- [ ] SFT
- [ ] RL
- [ ] 统一评测与最终报告

## 最近结果

- 8 卡 BF16 d6 smoke 在 step 50 保存后，从完整 checkpoint 恢复至 step 100。
- validation bpb：3.164625（step 0）→ 1.960774（step 50）→ 1.836842（step 100）。
- 稳态吞吐约 0.9M–1.0M token/s；峰值显存约 1.1 GiB/rank。
- step 50 与 step 100 均有模型、metadata 和 8 份 optimizer state。
- 两个 SwanLab 离线 run 已落盘，等待 L20 完成安全登录后同步。
- 对齐 `kernels==0.11.7` 并修复 Torch 2.6 graph-break/tuple 兼容后，FA3 GPU 回归 20 项通过。
- d24 正式 batch 探针稳态约 5.17 秒/step、峰值分配显存 17.9 GiB/rank，预计正式 Base 约 31–33 小时。
- 详细证据见 [logs/smoke-attempt-2.md](logs/smoke-attempt-2.md)。

## 不可变约束

- Git 操作只在 L20 执行。
- 正式训练使用 BF16，不启用 FP8。
- 未通过 8 卡 100-step 冒烟和 step 50 恢复验证，不启动 d24。
- 数据、完整日志、SwanLab 原始目录和 checkpoint 保存在 CPFS，不进入 Git。
- GitHub 保存配置、轻量指标、日志摘要、图表、哈希和报告。

完整方案见 [../../docs/L20_SWANLAB_EXPERIMENT_PLAN.md](../../docs/L20_SWANLAB_EXPERIMENT_PLAN.md)。
