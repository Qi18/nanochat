# NanoChat GPT、优化器与 Checkpoint 源码阅读

## 阅读目标

理解 d24 模型的结构、BF16 数据流、AdamW/Muon 参数分组、多卡通信和 checkpoint 恢复边界。

## 模型构造

`scripts.base_train.build_model_meta` 根据 depth 计算模型宽度：

```text
base_dim = depth * aspect_ratio
model_dim = ceil(base_dim / head_dim) * head_dim
num_heads = model_dim / head_dim
```

本实验 d24 的默认值：

```text
depth = 24
aspect_ratio = 64
head_dim = 128
model_dim = 1536
num_heads = 12
sequence_len = 2048
```

当前 Base 构造把 `n_kv_head` 设置为 `n_head`。源码支持 GQA，但本次 d24 配置并没有减少 KV heads。

## GPT 结构

当前实现的主要特征：

- RoPE，没有可学习位置 embedding。
- Q/K 在 RoPE 后做 RMSNorm，并分别乘 1.2。
- token embedding 与 `lm_head` 不共享权重。
- MLP 使用 `ReLU(x)^2`。
- Linear 无 bias，master weight 保持 FP32，前向时转换为 activation dtype。
- embedding 和 value embedding 在 BF16 模式下直接保存为 BF16。
- alternating value embedding，通过输入相关 gate 注入 V。
- `resid_lambdas` 调整逐层 residual。
- `x0_lambdas` 将初始 embedding 重新注入深层。
- smear 把前一个 token embedding 作为轻量 bigram 信息加入当前 token。
- backout 在最终归一化前减去中间层 residual。
- logits 转 FP32 后使用 softcap=15，再计算 cross entropy。

## Forward Tensor 流

```text
idx [B,T]
  -> wte
  -> BF16 + RMSNorm
  -> smear
  -> 24 × Transformer Block
       q/k/v: [B,T,H,D]
       attention
       residual
       ReLU² MLP
  -> backout + RMSNorm
  -> lm_head [B,T,V]
  -> FP32 softcap
  -> cross entropy
```

本实验指定 `window-pattern=L`。如果 FA3 未加载，PyTorch SDPA 可以直接走 full causal attention，避免 sliding-window 显式 mask 带来的低利用率。

## 参数分组

`GPT.setup_optimizer` 把参数分成两类：

### AdamW

- `lm_head`
- token embedding
- value embeddings
- residual/x0 scalars
- smear/backout 参数

这些组使用不同学习率、beta 和 weight decay。AdamW 学习率还按 `1/sqrt(d_model/768)` 缩放。

### Muon

Transformer matrix 参数按 shape 分组，使用 Muon 更新。实现包含：

- momentum
- Polar Express 正交化
- MuonEq row equilibration
- Muon+ Frobenius norm 校准
- 方差缩放
- cautious weight decay

## 多卡通信

NanoChat 没有把模型包进 PyTorch DDP。多卡同步在 `MuonAdamW` 内部完成：

- 小 AdamW 参数使用 all-reduce。
- 大 AdamW 参数使用 reduce-scatter、分片更新、all-gather。
- Muon 同 shape 参数先 stack，再 reduce-scatter。
- 每个 rank 只维护自己负责的 optimizer state，属于 ZeRO-2 风格。
- 通信分为启动 reduce、等待并计算/启动 gather、等待 gather 三个阶段，用于重叠通信和计算。

因此 NCCL 不是外围能力，而是 optimizer step 的核心依赖。

## Checkpoint

每个 Base checkpoint 包含：

```text
model_<step>.pt
meta_<step>.json
optim_<step>_rank0.pt
...
optim_<step>_rank7.pt
```

- 模型和 metadata 只由 rank 0 保存。
- optimizer state 每个 rank 单独保存。
- metadata 包含模型配置、用户参数、batch 配置、dataloader state 和训练 loop state。
- 恢复训练时，8 个 optimizer shard 缺一不可。

本次 50→100 step 冒烟需要同时验证：

1. `model_000050.pt` 可加载；
2. 8 个 optimizer shard 都存在；
3. metadata 中的 step、batch 和 dataloader state 完整；
4. 恢复后能够生成 `model_000100.pt`。

## 训练 horizon

当没有显式 `num_iterations` 时，Base 训练根据：

```text
target_tokens = target_param_data_ratio × scaling_params
num_iterations = target_tokens // total_batch_size
```

本实验 d24 设置 ratio=8、total batch=524288。实际 scaling params、iterations 和总 token 以正式启动日志为准。

## 冒烟验证结果

- Torch 2.6 成功编译当前 GPT 和 Muon 路径，首次 step 编译约 25.6 秒；恢复进程命中部分缓存，首 step 约 6.4 秒。
- d6、单卡 batch 2、8 卡的稳态吞吐约 0.9M–1.0M token/s，峰值显存约 1.1 GiB/rank。
- L20 未加载 FA3，实际使用 PyTorch SDPA；MFU 因源码尚未定义 L20 峰值而显示 0，不能据此判断利用率。
- 模型文件约 184.5 MiB，每份 optimizer shard 约 42.2 MiB。
- dataloader state 从 step 50 的 `epoch=1,pq_idx=0,rg_idx=8` 前进到 step 100 的 `epoch=1,pq_idx=0,rg_idx=24`。
- validation bpb 从 3.164625 降至 1.836842，所有已记录 loss/BPB 均为有限值。
- `train/epoch` 原为复合字符串，SwanLab 0.9.2 拒绝该指标；现拆为 `train/epoch`、`train/pq_idx`、`train/rg_idx` 三个整数指标，并通过真实离线 run 验证。
