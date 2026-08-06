# NanoChat 数据、Tokenizer 与 DataLoader 源码阅读

## 阅读目标

理解 ClimbMix parquet 文件如何经过 BPE 训练、tokenizer 持久化和 BOS-aligned best-fit packing，最终生成 Base 训练使用的 `inputs` 与 `targets`。

## 调用链

```text
nanochat.dataset
  -> base_data_climbmix/shard_*.parquet
  -> scripts.tok_train
  -> RustBPETokenizer.train_from_iterator
  -> tokenizer/tokenizer.pkl + token_bytes.pt
  -> tokenizing_distributed_data_loader_with_state_bos_bestfit
  -> inputs[B, T] + targets[B, T] + resume_state
```

## Dataset

### 源码事实

- `nanochat.dataset` 使用编号 parquet shard，`shard_06542.parquet` 固定作为 validation shard。
- `-n 170` 表示下载 170 个训练 shard，并额外下载一个 validation shard。
- 下载先写入 `.tmp`，完成后通过 rename 变为正式 parquet，避免 dataloader 读到半文件。
- 单 shard 最多重试 5 次，重试间隔使用指数退避。
- L20 无法直连代码中原本写死的 `huggingface.co`，本实验加入 `NANOCHAT_DATASET_BASE_URL`，显式切换到 `hf-mirror.com`。

## Tokenizer

### 训练与推理实现

- 训练端使用 `rustbpe.Tokenizer`，推理端构造 `tiktoken.Encoding`。
- 默认词表大小为 32768，其中包含普通 byte/BPE token 和 9 个对话特殊 token。
- 数字正则使用 `\p{N}{1,2}`，即数字倾向按一到两位分组。
- 每篇训练文档默认最多取 10000 字符，Tokenizer 总训练上限为 20 亿字符。
- `tokenizer.pkl` 保存 tiktoken encoding。
- `token_bytes.pt` 保存每个 token 对应的原始字节数，用于计算与词表大小相对无关的 BPB。

### 特殊 token

- `<|bos|>`
- user/assistant start/end
- python start/end
- output start/end

这些 token 既服务于 Base 数据的文档边界，也服务于 SFT 和 RL 的对话渲染。

## DataLoader

### BOS-aligned best-fit

对每个训练行创建长度 `T + 1` 的 buffer：

1. 从文档 buffer 中选择能完整放入剩余空间的最长文档。
2. 如果没有文档可以完整放入，选择最短文档并裁剪到正好填满。
3. `row[:, :-1]` 成为 inputs，`row[:, 1:]` 成为 targets。

因此最终 Tensor shape 为：

```text
row_buffer: [B, T + 1]
inputs:     [B, T]
targets:    [B, T]
```

该设计没有 padding，token 利用率为 100%，代价是源码注释所说大约 35% token 因文档裁剪被丢弃。

### DDP 与恢复

- 每个 rank 从不同 row group 开始，步长为 world size。
- checkpoint 保存 `pq_idx`、`rg_idx` 和 `epoch`。
- 恢复时从上次 row group 的下一个 DDP 对齐位置继续，避免简单重复上一批数据。
- 这是 row-group 粒度的近似恢复，不保存 Python 文档 buffer 的精确内部状态。

## 与本次实验的关系

- 数据固定保存到 `/data/cache/nanochat/base_data_climbmix`。
- Tokenizer 固定保存到 `/data/cache/nanochat/tokenizer`。
- GitHub 不保存 parquet 或 tokenizer 二进制，只记录 shard 名称、大小、列表哈希和 tokenizer 文件 SHA256。
- checkpoint 恢复测试同时验证模型、optimizer 与 dataloader state。

## 实验观察

- Hugging Face 官方地址在 L20 连接超时。
- hf-mirror 对同一 shard 返回 HTTP 200 并可完整读取。
- rustbpe 0.1.0 与当前 Python 3.10 环境可正常导入。
- 实际 shard 数量、Tokenizer 训练耗时和压缩率在数据阶段结束后回填。

## 待验证

- 8 个初始 shard 是否足够稳定覆盖 20 亿字符的 tokenizer 上限。
- Tokenizer 实际训练用到的字符数量和耗时。
- train/validation 压缩率与 GPT-2、GPT-4 tokenizer 的差异。
- 50→100 step 恢复后 dataloader state 是否按预期前进。
