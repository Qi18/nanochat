# NanoChat 8×L20 SwanLab 实验与源码阅读方案

> 实验编号：`l20-d24-swanlab-20260806`
> 实验状态：规划中，尚未启动正式训练
> 代码仓库：`Qi18/nanochat`
> 集成基线分支：`experiment/swanlab-l20`
> 计划实验分支：`experiment/l20-d24-swanlab-20260806`

## 1. 背景

本实验在 8 张 NVIDIA L20 上完成 NanoChat 的 Base 预训练、SFT、RL 和统一评测，并使用 SwanLab 记录训练过程。实验期间同步阅读 NanoChat 源码，把源码事实、实验观察和个人分析整理为可发布的博客文章。

本次实验首先建立“当前代码版本的可复现基线”，不立即修改模型结构或训练配方。此前实验基于较旧的代码版本，当前上游已经包含训练、评测及 BPB 计算相关修复，因此历史结果只作为参考，不作为严格同口径对照。

## 2. 总体目标

1. 在当前代码上复现一条完整的 Base → SFT → RL 训练链路。
2. 使用 SwanLab 记录配置、指标、日志、硬件环境和代码版本。
3. 比较 SFT、RL 最优 checkpoint 和 RL 最终 checkpoint，确认 RL 的收益与能力退化。
4. 将实验方案、启动命令、配置、指标、日志摘要、图表和最终报告上传 GitHub。
5. 在训练期间完成 NanoChat 核心源码阅读，并形成一组可追溯的博客文章。
6. 沉淀一套后续实验可以直接复用的目录、脚本和报告模板。

## 3. 执行边界

- 所有分支创建、提交和 GitHub 推送均在 L20 上执行。
- 本地工作区不提交、不推送，也不作为实验代码来源。
- 数据集、完整控制台日志、SwanLab 原始目录和模型 checkpoint 不提交 Git。
- 大文件保存在 `/data` 的 CPFS；GitHub 只保存路径、大小、哈希和必要的轻量结果。
- 不提交 SwanLab Token、GitHub Token、SSH 密钥、内部地址或其他凭证。
- 未通过 8 卡冒烟和 checkpoint 恢复验证，不启动正式 d24 训练。

## 4. 三线并行工作流

实验按照三条工作线并行推进：

1. **训练实验线**：环境检查、数据准备、Base、SFT、RL 和统一评测。
2. **源码阅读线**：围绕当前训练阶段阅读入口脚本、核心模块和评测实现。
3. **内容输出线**：将 SwanLab 数据、GitHub 证据和源码笔记整理为博客草稿及最终报告。

每个阶段必须使用同一个实验编号，并记录：

- 训练 Git commit
- SwanLab Run ID 和 URL
- 配置文件和真实启动命令
- checkpoint 标识
- 对应源码笔记
- 对应博客文章或草稿

## 5. 实验产物流向

```text
L20 实验分支
  ├─ 8×L20 训练 ────────────> SwanLab 指标、日志和运行环境
  ├─ CPFS 数据/checkpoint ──> checkpoint 清单、大小和 SHA256
  ├─ 源码阅读 ──────────────> source-notes/*.md
  └─ 阶段结果 ──────────────> blog/drafts/*.md
                                │
                                └─ L20 提交并推送 GitHub
```

## 6. Git 分支与提交策略

### 6.1 分支

正式实验开始时，仅在 L20 执行：

```text
experiment/swanlab-l20
└── experiment/l20-d24-swanlab-20260806
```

训练前先提交完整实验配置，保证训练使用的 commit 不再漂移。训练中不合并上游更新；如必须修复代码，单独提交修复并在报告中记录旧、新 commit 及受影响的阶段。

### 6.2 阶段提交

建议提交节奏：

1. `docs: add L20 SwanLab experiment plan`
2. `experiment: add preflight and run configuration`
3. `experiment: record environment and smoke results`
4. `experiment: record base training results`
5. `experiment: record SFT results`
6. `experiment: record RL results`
7. `docs: publish final experiment report`

每次只显式暂存本阶段文件，不使用不加区分的全目录提交。

## 7. GitHub 实验目录

```text
experiments/l20-d24-swanlab-20260806/
├── README.md
├── PLAN.md
├── config/
│   ├── environment.txt
│   ├── provenance.json
│   ├── base.env
│   ├── sft.env
│   └── rl.env
├── scripts/
│   ├── 00_preflight.sh
│   ├── 01_prepare_data.sh
│   ├── 02_smoke.sh
│   ├── 03_base_train.sh
│   ├── 04_base_eval.sh
│   ├── 05_sft.sh
│   ├── 06_sft_eval.sh
│   ├── 07_rl.sh
│   └── 08_compare.sh
├── metrics/
│   ├── base.json
│   ├── sft.json
│   ├── rl.json
│   └── comparison.csv
├── logs/
│   ├── README.md
│   └── *.log.gz
├── figures/
├── artifacts/
│   └── checkpoints.json
├── source-notes/
│   ├── 01-execution-pipeline.md
│   ├── 02-dataset-tokenizer.md
│   ├── 03-gpt-architecture.md
│   ├── 04-training-engine.md
│   ├── 05-attention.md
│   ├── 06-evaluation.md
│   ├── 07-sft.md
│   └── 08-rl.md
├── blog/
│   ├── drafts/
│   ├── published/
│   └── publish-manifest.json
└── FINAL_REPORT.md
```

`logs/` 只保存便于审查的压缩阶段日志或日志摘要。多卡完整原始日志保留在 CPFS，并在 `logs/README.md` 中记录路径和哈希。

## 8. L20 环境与存储

已知目标环境：

- GPU：8×NVIDIA L20，每卡约 46 GB
- 共享存储：`/data`，CPFS
- 仓库：`/data/projects/nanochat`
- 推荐缓存根目录：`/data/cache/nanochat`

统一环境变量：

```bash
export NANOCHAT_BASE_DIR=/data/cache/nanochat
export HF_HOME=/data/cache/huggingface
export SWANLAB_LOG_DIR=/data/cache/swanlab
export HF_ENDPOINT=https://hf-mirror.com
export NANOCHAT_DTYPE=bfloat16
```

正式训练前记录：

- `git rev-parse HEAD`
- `git status --short`
- `nvidia-smi`
- GPU 数量和显存
- Python、PyTorch、CUDA、NCCL 版本
- `pip freeze` 或 `uv pip freeze`
- `/data` 和 `/dev/shm` 容量
- tokenizer 与数据清单哈希

不能只检查 `torch.cuda.is_available()`，还要实际完成 CUDA Tensor 运算和 8 卡通信测试。

## 9. SwanLab 记录设计

### 9.1 项目与分组

```text
project: nanochat-lab
group: l20-d24-swanlab-20260806
```

建议 Run：

| Run 名称 | job_type | 内容 |
|---|---|---|
| `l20-d24-env-smoke` | `smoke` | 环境、CUDA、NCCL 和恢复测试 |
| `l20-d24-base` | `train` | Base 正式训练 |
| `l20-d24-base-eval` | `eval` | Base BPB/CORE 评测 |
| `l20-d24-sft` | `train` | SFT |
| `l20-d24-sft-eval` | `eval` | SFT 统一评测 |
| `l20-d24-rl` | `train` | RL |
| `l20-d24-rl-best-eval` | `eval` | RL 最优 checkpoint 评测 |
| `l20-d24-rl-final-eval` | `eval` | RL 最终 checkpoint 评测 |

### 9.2 每个 Run 的最小记录集合

- 实验编号、Git branch、commit、dirty 状态
- GPU、驱动、Python、PyTorch、CUDA 和依赖版本
- 随机种子
- 数据 shard 数量、tokenizer 哈希
- 模型深度、参数量、序列长度和 window pattern
- dtype、单卡 batch、总 batch 和梯度累积
- loss、BPB、学习率、梯度范数
- tokens/sec、MFU、显存和阶段耗时
- checkpoint step、路径、大小和 SHA256
- 评测任务及结果

现有集成通过 `swanlab.sync_wandb(..., wandb_run=False)` 复用训练脚本已有的 W&B 日志调用，关闭 W&B 上传，只写入 SwanLab。

### 9.3 异常恢复

SwanLab 在线连接失败不应终止训练：

1. 保留本地 SwanLab 日志目录。
2. 切换或降级为离线记录。
3. 训练完成或网络恢复后执行 `swanlab sync`。
4. 保存原 Run ID，尽量同步或恢复到同一个实验记录。

## 10. 实验阶段与源码阅读路线

| 阶段 | 实验任务 | 阅读源码 | 重点问题 | 内容输出 |
|---|---|---|---|---|
| 环境准备 | 构建环境、验证 8 卡 | `runs/speedrun.sh`、`runs/runcpu.sh`、`pyproject.toml`、`nanochat/execution.py` | 完整流水线如何启动，多卡进程如何组织 | 环境与流水线文章 |
| 数据准备 | 下载数据、训练 tokenizer | `nanochat/dataset.py`、`nanochat/tokenizer.py`、`nanochat/dataloader.py`、`scripts/tok_train.py` | 数据如何变为训练 batch | 数据与 Tokenizer 文章 |
| Base 冒烟 | 100 steps、保存与恢复 | `nanochat/checkpoint_manager.py`、`nanochat/common.py` | checkpoint 如何保存和恢复 | 冒烟记录 |
| Base 正式训练 | d24 多卡预训练 | `scripts/base_train.py`、`nanochat/gpt.py`、`nanochat/engine.py`、`nanochat/optim.py` | 模型结构、前向、反向和优化器 | GPT 预训练文章 |
| Attention | 验证 BF16 和全注意力配置 | `nanochat/flash_attention.py`、`nanochat/fp8.py`、`nanochat/gpt.py` | L20 为什么使用 BF16，window pattern 如何影响计算 | Attention 与精度文章 |
| Base 评测 | BPB、CORE、样例生成 | `scripts/base_eval.py`、`nanochat/loss_eval.py`、`nanochat/core_eval.py`、`tasks/*.py` | 指标如何定义和聚合 | NanoChat 评测体系文章 |
| SFT | 对话微调 | `scripts/chat_sft.py`、`nanochat/tokenizer.py`、`nanochat/dataloader.py` | 对话渲染、packing、mask 和有效 target | SFT 源码文章 |
| RL | 强化学习和 checkpoint 选择 | `scripts/chat_rl.py`、`nanochat/engine.py` | rollout、奖励、更新和后期退化 | RL 源码文章 |
| 统一评测 | SFT/RL best/RL final 对比 | `scripts/chat_eval.py`、`tasks/arc.py`、`tasks/gsm8k.py`、`tasks/mmlu.py`、`tasks/humaneval.py` | RL 改善了什么，又损失了什么 | 最终实验报告 |
| 追踪实现 | 核对记录链路 | `nanochat/experiment_tracking.py`、`docs/SWANLAB.md` | 指标如何同步到 SwanLab | SwanLab 接入文章 |

## 11. 源码阅读方法

每篇源码笔记使用统一模板：

```markdown
# 模块名称

## 阅读目标
## 入口与调用链
## 核心数据结构和 Tensor shape
## 关键源码
## 对应实验与 SwanLab Run
## 源码事实
## 实验观察
## 我的理解
## 未解决问题
```

写作时区分三种信息：

1. **源码事实**：可以定位到固定 Git commit 的文件和代码位置。
2. **实验观察**：可以定位到 SwanLab Run、指标文件和 checkpoint。
3. **个人分析**：明确标记为解释、推测或待验证假设。

源码引用使用训练 commit 对应的 GitHub permalink，不引用会随分支变化的普通链接。博客只摘录必要的小段代码，其余通过链接定位。

## 12. 博客输出计划

计划文章：

1. 《NanoChat 实验流水线与 8×L20 环境搭建》
2. 《NanoChat 的数据集、Tokenizer 与 DataLoader》
3. 《从源码理解 NanoChat GPT 预训练》
4. 《NanoChat Attention 与 L20 精度选择》
5. 《NanoChat 的 BPB、CORE 与 Benchmark》
6. 《NanoChat SFT 数据流与训练实现》
7. 《NanoChat RL：收益、退化与 Checkpoint 选择》
8. 《8×L20 完整训练 NanoChat：SwanLab 实验报告》

### 12.1 发布节奏

- 下载数据期间：整理流水线和数据模块草稿。
- Base 正式训练期间：深入阅读 GPT、Attention、Engine 和 Optimizer。
- SFT 完成后：发布 SFT 源码与曲线分析。
- RL 完成后：发布 RL 机制与 checkpoint 对比。
- 全量评测完成后：发布最终实验报告。

中间文章标记为“源码阅读”或“实验日志”，最终报告才给出完整结论。

### 12.2 文章证据要求

每篇文章必须包含：

- 实验编号
- NanoChat Git commit
- 固定 commit 的源码链接
- 对应 SwanLab Run URL
- 使用的指标 JSON/CSV 或图表来源
- 测试硬件和关键配置
- 发布日期

如果 SwanLab 项目不是公开的，同时在 GitHub 保存曲线图片和轻量指标，避免文章只能依赖私有链接。

`blog/publish-manifest.json` 记录文章状态：

```json
{
  "experiment_id": "l20-d24-swanlab-20260806",
  "git_commit": "<training-commit>",
  "articles": [
    {
      "title": "<title>",
      "source": "blog/published/<article>.md",
      "url": "<published-url>",
      "published_at": "<timestamp>",
      "swanlab_runs": ["<run-id>"]
    }
  ]
}
```

博客平台未确定前，统一生成标准 Markdown，不阻塞实验。确定平台后再补充发布脚本或同步流程。

## 13. 分阶段执行方案

### 阶段 0：代码与环境固化

1. 从 `experiment/swanlab-l20` 创建实验分支。
2. 写入实验目录、配置和运行脚本。
3. 安装项目与 tracking 依赖。
4. 记录 Git、Python、PyTorch、CUDA、NCCL 和 GPU 环境。
5. 验证 `/data` 读写、外网、Hugging Face 镜像、SwanLab 和 GitHub SSH。
6. 提交实验配置，得到不可变的训练 commit。

### 阶段 1：数据与 Tokenizer

1. 将数据下载到 CPFS。
2. 训练或准备 tokenizer。
3. 记录 shard 数量、文件清单和哈希。
4. 检查一个真实 batch 的 shape、padding 和有效 target 数量。
5. 完成数据与 tokenizer 源码笔记。

### 阶段 2：8 卡冒烟与恢复测试

先运行 d6 或 d12、约 100 steps：

- 验证 DDP/NCCL。
- 验证 loss、BPB 和 SwanLab 曲线。
- 验证 checkpoint 保存。
- 主动停止后从 checkpoint 恢复。
- 记录显存、tokens/sec 和预估正式训练时长。

以下条件全部满足才进入正式训练：

- loss/BPB 有限且趋势合理。
- 8 个 rank 均正常退出。
- checkpoint 可以加载并继续训练。
- SwanLab Run 包含配置、日志和指标。
- Git commit 和运行代码一致。

### 阶段 3：d24 Base 训练

起始配置建议：

```text
depth=24
target-param-data-ratio=8
device-batch-size=2
total-batch-size=524288
window-pattern=L
dtype=bfloat16
fp8=disabled
eval-every=250
save-every=1000
```

最终 batch、梯度累积和评测频率以冒烟结果为准。不能直接照搬面向 H100 的 FP8 配置。

### 阶段 4：Base 评测

记录：

- validation BPB
- CORE 总分和子任务
- 参数量和总训练 token
- 实际耗时、吞吐和峰值显存
- 最终 checkpoint 路径、大小和 SHA256
- 固定 prompts 的生成样例

### 阶段 5：SFT

重点监控：

- train loss 和 validation BPB
- `valid_targets`
- padding 比例
- 最优和最终 checkpoint
- 固定对话样例

如出现 `valid_targets=0`，优先检查全 padding batch、对话渲染和 packing，不通过修改学习率掩盖数据问题。

### 阶段 6：RL

至少保留并评测：

- SFT best
- RL 曲线选出的 best
- RL final

不能只评最后一步。按照 SwanLab 曲线和中间评测共同确定 RL best，并记录选择依据。

### 阶段 7：统一评测

对三个候选 checkpoint 使用相同配置评测：

- ARC-Easy
- ARC-Challenge
- MMLU
- GSM8K
- HumanEval
- ChatCORE
- 固定人工 prompts
- 历史失败样例，包括多位数乘法

最终报告必须分别说明通用能力、数学推理、代码能力和对话能力的变化。

## 14. Checkpoint 管理

建议保留：

- Base：最终 checkpoint + 一个可恢复的中间 checkpoint
- SFT：best + final
- RL：best + final，必要时保留关键中间点

GitHub 中的 `artifacts/checkpoints.json` 记录：

```json
{
  "stage": "rl",
  "step": 240,
  "role": "best",
  "path": "/data/cache/nanochat/.../model_000240.pt",
  "size_bytes": 0,
  "sha256": "<sha256>",
  "git_commit": "<commit>",
  "swanlab_run_id": "<run-id>"
}
```

如需长期保存模型，后续同步 OSS；不把大型权重直接放入普通 Git 历史。

## 15. 最终报告

`FINAL_REPORT.md` 至少包含：

1. 实验目标和结论摘要
2. Git 分支、训练 commit 和代码变更
3. L20 硬件及软件环境
4. 数据与 tokenizer 信息
5. 真实训练命令和配置
6. SwanLab Run 列表与链接
7. Base、SFT、RL 曲线和核心指标
8. SFT、RL best、RL final 对比表
9. 与历史实验的差异及指标口径说明
10. 异常、恢复和失败记录
11. checkpoint 清单和 SHA256
12. 总耗时和各阶段耗时
13. 源码理解如何解释实验结果
14. 博客文章索引
15. 下一轮改进建议

## 16. 停止条件

出现以下任一情况暂停当前阶段并保留现场：

- loss 或 BPB 为 NaN/Inf
- `valid_targets=0`
- CUDA OOM、NCCL 错误持续出现
- checkpoint 无法保存或恢复
- `/data` 空间或读写异常
- 实际运行代码与记录的 Git commit 不一致
- 数据或 tokenizer 哈希发生意外变化
- SwanLab 没有记录核心配置和指标，且离线日志也不存在

SwanLab 单纯断网不属于停止正式训练的条件，只要离线记录完整并能够后续同步。

## 17. 验收标准

### 17.1 实验验收

- Base、SFT 和 RL 均有完整 SwanLab Run。
- Base 训练无 NaN/Inf，checkpoint 可恢复。
- SFT 无全 padding batch，能够确定 best checkpoint。
- RL 完成 SFT、RL best 和 RL final 的同配置对比。
- GitHub 包含方案、配置、命令、指标、图表、checkpoint 清单和最终报告。
- GitHub 中没有凭证、数据集或模型大文件。

### 17.2 源码阅读验收

- 每个训练阶段至少完成一篇源码笔记。
- 入口、调用链、关键数据结构和 Tensor shape 有明确记录。
- 源码结论引用固定训练 commit。
- 能用源码解释主要训练指标、异常和 checkpoint 选择。

### 17.3 博客验收

- 文章源码以 Markdown 保存到 GitHub。
- 文章中的指标与 SwanLab、GitHub 报告一致。
- 已发布 URL 回填 `publish-manifest.json`。
- 文章明确区分源码事实、实验观察和个人分析。
- 内容不包含密钥或未经脱敏的信息。

## 18. 时间预算

| 阶段 | 预计耗时 |
|---|---:|
| 环境和数据准备 | 1～2 小时 |
| 冒烟及恢复验证 | 0.5～1 小时 |
| d24 Base | 约 12～15 小时，冒烟后重算 |
| SFT | 约 1 小时 |
| RL 和全量评测 | 约 2～4 小时 |
| 阶段报告与最终整理 | 0.5～1 小时 |

源码阅读和博客草稿主要在数据下载、正式训练和评测等待期间进行，不额外占用 GPU。整体实验预计 16～23 小时，最终以冒烟测试实测吞吐为准。

## 19. 启动前检查清单

- [ ] L20 仓库位于 `experiment/l20-d24-swanlab-20260806`
- [ ] 工作区干净，训练 commit 已记录
- [ ] 8 张 L20 全部可见并通过 CUDA/NCCL 测试
- [ ] `/data`、`/dev/shm` 容量和读写正常
- [ ] Hugging Face 镜像可用
- [ ] SwanLab 在线或离线记录可用
- [ ] GitHub SSH 推送链路可用
- [ ] 环境快照已保存
- [ ] 数据与 tokenizer 清单、哈希已保存
- [ ] 8 卡 100-step 冒烟通过
- [ ] checkpoint 保存和恢复通过
- [ ] 博客平台或 Markdown 暂存策略已确定

完成以上检查后，才启动正式 d24 Base 训练。
