# NanoChat 执行流水线源码阅读

## 阅读目标

从仓库真实入口理解 NanoChat 如何把环境准备、数据、Tokenizer、Base 训练、评测和 SFT 串成一条可执行流水线。

## 入口与调用链

`runs/speedrun.sh` 的主链路是：

```text
nanochat.dataset
  -> scripts.tok_train
  -> scripts.tok_eval
  -> torchrun scripts.base_train
  -> torchrun scripts.base_eval
  -> torchrun scripts.chat_sft
  -> torchrun scripts.chat_eval
```

## 源码事实

1. `runs/speedrun.sh` 面向 8×H100，默认使用 d24、单卡 batch 16 和 FP8。
2. `scripts.base_train` 通过 `torchrun` 环境变量初始化 DDP，并在 rank 0 负责日志与 checkpoint 元数据。
3. 训练脚本使用 `NANOCHAT_DTYPE` 选择计算精度；SM 80+ 默认 BF16。
4. 未加载 FA3 时，`nanochat.flash_attention` 回退到 PyTorch SDPA。
5. 数据下载地址原本写死为 Hugging Face 官方域名，L20 直连超时，因此本实验增加显式镜像环境变量。

## L20 实验映射

- H100 的 FP8 配方不直接复制到 L20。
- 本实验使用 BF16、`window-pattern=L`、较小单卡 batch 和梯度累积。
- 正式训练前先进行 8 卡 NCCL、50-step 保存和 50→100 恢复验证。
- SwanLab 通过现有 W&B 日志调用收集 Base/SFT/RL 指标，W&B 不上传。

## 已验证观察

- 8 张 L20 均可执行 CUDA Tensor 运算。
- 8 个 rank 的 NCCL all-reduce 结果均为 36。
- Hugging Face 官方数据地址在 L20 超时，`hf-mirror.com` 可下载同一 shard。
- 隔离环境中的 SwanLab、W&B、rustbpe 和 kernels 可导入。

## 待继续阅读

- `nanochat.execution` 如何管理脚本参数和进程。
- `nanochat.checkpoint_manager` 的分片 optimizer 状态。
- `base_train` 中 batch、梯度累积与训练 horizon 的推导。
- SwanLab Run 与 checkpoint manifest 如何建立双向索引。
