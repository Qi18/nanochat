# l20-d24-swanlab-20260806

NanoChat 在 8×NVIDIA L20 上的当前代码基线复现实验，使用 SwanLab 记录 Base、SFT、RL 和评测过程，并同步产出源码阅读笔记与博客草稿。

## 当前状态

- [x] 创建 L20 独立实验分支
- [x] ACK API、真实 SSH、8×L20、CPFS 和共享内存检查
- [x] PyTorch CUDA Tensor 与 8 卡 NCCL all-reduce
- [x] SwanLab/W&B/rustbpe/kernels 隔离环境
- [x] Hugging Face 直连与镜像验证
- [ ] 固化 provenance
- [ ] 数据与 tokenizer
- [ ] 100-step checkpoint 恢复冒烟
- [ ] d24 Base
- [ ] Base Eval
- [ ] SFT
- [ ] RL
- [ ] 统一评测与最终报告

## 不可变约束

- Git 操作只在 L20 执行。
- 正式训练使用 BF16，不启用 FP8。
- 未通过 8 卡 100-step 冒烟和 step 50 恢复验证，不启动 d24。
- 数据、完整日志、SwanLab 原始目录和 checkpoint 保存在 CPFS，不进入 Git。
- GitHub 保存配置、轻量指标、日志摘要、图表、哈希和报告。

完整方案见 [../../docs/L20_SWANLAB_EXPERIMENT_PLAN.md](../../docs/L20_SWANLAB_EXPERIMENT_PLAN.md)。
