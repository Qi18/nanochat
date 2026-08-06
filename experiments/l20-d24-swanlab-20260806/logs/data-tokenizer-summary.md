# 数据与 Tokenizer 阶段摘要

## 数据

- 来源：ClimbMix-400B shuffle，经 hf-mirror 下载
- train shard：170
- validation shard：1
- 总 shard：171
- CPFS 占用：约 15GB
- 缺失 shard：0
- 残留 `.tmp`：0
- shard 名称与大小列表 SHA256：`155a9c73c3fe2341373b4bfd9795cd530f13ea0dab071cf7401d71fd2ae841cb`

## Tokenizer

- vocab size：32768
- merges：32503
- 训练耗时：53.30 秒
- `tokenizer.pkl`：412105 bytes
- `tokenizer.pkl` SHA256：`387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c`
- `token_bytes.pt`：132252 bytes
- `token_bytes.pt` SHA256：`ea2ed770d0f77f8e8c82477bbcbabb8056c3d3a72a7ed29a599726838835aa8a`

## 压缩率观察

- 相对 GPT-2：ClimbMix train +1.4%，validation +1.2%，code +31.2%。
- 相对 GPT-4：ClimbMix train -1.8%，validation -2.2%，Korean 差距最大。

## CPFS 日志

| 日志 | bytes | SHA256 |
|---|---:|---|
| `dataset-initial.log` | 882 | `2a07ae1a0f4351fbc6cfd7b139302ecd422f56baf827e506b312ec5cd81e5e13` |
| `dataset-full.log` | 6381 | `48d0e4ff4a0b9ca175d6aecbb95b3da758193e49b5bbdf78b1fc7ff10601777a` |
| `tokenizer-train.log` | 16069 | `c9e40850c6881d432a8cc95206ce084e617bec4fa2dd9c356955d9e34852d786` |
| `tokenizer-eval.log` | 2473 | `d665a7131c8b4c21792a10a6a269139120563cd8e2273fe67f41969bc269955a` |
| `data-supervisor.log` | 19551 | `556dad69562acc5dd40ce34349cfd939d1fa0a17b1ee0739b4783b70806515b5` |

日志根目录：

`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/`
