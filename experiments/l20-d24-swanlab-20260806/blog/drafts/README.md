# 博客草稿

实验期间的博客统一先以 Markdown 草稿保存在这里。每篇文章必须标注：

- 实验编号
- 固定 Git commit
- 源码 permalink
- SwanLab Run ID/URL
- 指标来源
- 源码事实、实验观察和个人分析的边界

博客平台确定后，再把发布 URL 回填到 `../publish-manifest.json`。

## 草稿索引

1. [`01-nanochat-data-tokenizer.md`](01-nanochat-data-tokenizer.md)：数据集、Tokenizer 与 DataLoader。
2. [`02-nanochat-expandable-segments.md`](02-nanochat-expandable-segments.md)：CUDA allocator 的 expandable segments。
3. [`03-nanochat-base-train-structure.md`](03-nanochat-base-train-structure.md)：`base_train.py` 的完整控制流与 d24 实验映射。
