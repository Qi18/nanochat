# NanoChat 的数据集、Tokenizer 与 DataLoader

> 状态：实验中草稿
> 实验：`l20-d24-swanlab-20260806`
> 代码：`experiment/l20-d24-swanlab-20260806`

训练一个语言模型时，“把文本交给模型”这句话隐藏了三层完全不同的工作：

1. 从数据源获得稳定、可恢复的数据文件；
2. 把字符串编码为 token；
3. 把长度不同的文档装进固定 shape 的训练 Tensor。

NanoChat 分别用 `dataset.py`、`tokenizer.py` 和 `dataloader.py` 完成这三件事。

## 1. ClimbMix shard

NanoChat 把预训练数据组织为编号 parquet shard，并把最后一个固定 shard 留作 validation。下载器使用临时文件加 rename，避免训练进程读到尚未下载完成的数据。

本次 L20 实验还暴露了一个环境问题：代码中的官方 Hugging Face 地址无法直连，而 `HF_ENDPOINT` 对写死的 requests URL 不生效。因此实验分支增加了显式的 `NANOCHAT_DATASET_BASE_URL`，使用 hf-mirror，同时保留官方地址作为默认值。

## 2. rustbpe 训练，tiktoken 推理

NanoChat 没有用同一个库承担全部 tokenizer 工作：

- rustbpe 负责从文本迭代器训练 merge；
- tiktoken 负责高效编码和解码；
- 9 个特殊 token 在 BPE 训练后追加。

默认词表为 32768。数字正则按一到两位切分，这是小词表下对数字 token 成本的一次显式取舍。

## 3. 为什么还要保存 token_bytes

普通交叉熵会受到 tokenizer 词表和切分方式影响。NanoChat 为每个 token 保存原始字节数，在 validation 上计算 bits per byte，使不同 tokenizer 下的结果更容易比较。

这里有一个容易忽略的细节：不能先把单个 token 解码成 Unicode 字符串再计算字节数，因为某些 token 本身不是合法的独立 UTF-8。当前实现直接读取 token 的原始 bytes。

## 4. Best-fit packing

DataLoader 为每行准备 `T + 1` 个 token：

```text
row_buffer: [B, T + 1]
inputs:     [B, T]
targets:    [B, T]
```

它优先挑选能完整放入剩余空间的最长文档；没有文档能放下时，就裁剪一个文档填满剩余位置。这样没有 padding，但会丢弃一部分被裁剪的文档 token。

## 5. DDP 如何分数据

不同 rank 按 row group 交错读取 parquet。checkpoint 保存 parquet 序号、row group 和 epoch，恢复时前进到下一个 DDP 对齐位置。

这能避免明显重复，但不是逐 token 的完全恢复，因为内存中的 best-fit 文档 buffer 没有进入 checkpoint。

## 6. 本次实验还要回填的数据

- 171 个 shard 的实际体积
- Tokenizer 训练耗时
- 词表文件 SHA256
- train/validation 压缩率
- checkpoint 恢复前后的 dataloader state

这些结果将来自 GitHub 的 `data_manifest.json`、训练 checkpoint 元数据和 SwanLab Run，而不是手工估算。
