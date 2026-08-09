# NanoChat 源码阅读：base_train.py 第一行“正经代码”为什么是一个环境变量

> 状态：实验中草稿
> 实验：`l20-d24-swanlab-20260806`
> 代码：`experiment/l20-d24-swanlab-20260806`
> 源码位置：`scripts/base_train.py` L14-L15

打开 `scripts/base_train.py`，跳过 docstring 之后的第一行“正经代码”不是模型、不是数据，而是一个环境变量：

```python
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

这一行是给 PyTorch CUDA 显存分配器上的一道“防碎片化 OOM 保险”。

## 1. 默认分配器的问题：碎片化

PyTorch 默认的 caching allocator 通过 `cudaMalloc` 申请多个固定大小、互相隔离的 segment，张量在 segment 内部切分。训练中张量不断分配和释放，每个 segment 内部会留下大小不一的空洞：

```text
GPU 显存
┌─ Segment A (封死) ─┐ ┌─ Segment B (封死) ─┐ ┌─ Segment C ─┐
│ ██████ ░░░ ██████  │ │ ████ ░░░░░ ██████  │ │ ██████ ░░░  │  [空闲显存]
└────────────────────┘ └────────────────────┘ └─────────────┘
  ██ = 已用   ░░ = 段内空闲碎片
```

此时来一个需要 300MB 连续空间的新请求：所有碎片加起来够 300MB，但没有任何一块连续的 → `CUDA out of memory`，尽管 `nvidia-smi` 显示显存明明还有剩。

## 2. expandable_segments 的解法：段可以生长

开启 `expandable_segments:True` 后，分配器改用 CUDA 虚拟内存 API（`cuMemAddressReserve` / `cuMemMap`）：先预留一大段虚拟地址，物理显存页按需映射进来。segment 不再是封死的固定块，而是可以原地向后扩展：

```text
GPU 显存
┌─ 可扩展 Segment ──────────────────┐
│ ██████ ████████ ██████ ▒▒▒▒▒ ↔   │  [空闲显存可被吸收进段]
└───────────────────────────────────┘
  ▒▒ = 按需扩展的新映射页
```

同样的 300MB 请求：直接扩展段尾部，映射新的物理页 → 分配成功，碎片大幅减少。对预训练这种长时间、张量 shape 相对规律但仍有 eval/checkpoint 等间歇性大分配的负载，这个开关能显著降低跑到中途 OOM 的概率。

## 3. 为什么必须写在 import torch 之前

`base_train.py` 到 L24 才 `import torch`。分配器配置需要在 CUDA 分配器初始化之前设置才能生效，所以 L14-L15 先 `import os` 并写好环境变量，顺序不能颠倒。如果放到 `import torch` 之后再设置，这一行会被静默忽略。

## 4. 两个命名细节

- `PYTORCH_ALLOC_CONF` 是 PyTorch 较新版本引入的后端无关名称，旧名称为 `PYTORCH_CUDA_ALLOC_CONF`；两者在当前版本均可识别，但旧名会逐步废弃。
- 同一配置也可以在运行时通过 `torch.cuda.memory._set_allocator_settings` 设置，但环境变量方式最简单且不依赖初始化时序之外的 API，NanoChat 选择了前者。

以上第 1-3 节为源码与 PyTorch 文档事实，第 2 节结尾对预训练负载的判断以及第 4 节的取舍分析为个人理解。
