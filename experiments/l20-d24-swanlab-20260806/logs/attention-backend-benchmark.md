# FA3 与 PyTorch SDPA 训练速度对比

## 结论

在当前 `8×NVIDIA L20 + BF16 + Torch 2.6 + NanoChat d24` 软件栈上，关闭 NanoChat 的 FA3 路径、使用默认 PyTorch SDPA fallback 更快且更省显存：

- 合并两轮各 20 个有效 step 后，平均 step 时间从 5.17484 秒降至 5.01474 秒，降低 3.09%。
- 平均吞吐从 101,315 token/s 提升至 104,549 token/s，提高 3.19%。
- 中位 step 时间降低 3.80%，用于抵抗第二轮 SDPA 两个慢 step 的影响。
- 峰值 allocated memory 从 17,889.18 MiB/rank 降至 15,942.59 MiB/rank，减少 1,946.58 MiB/rank（10.88%）。

因此，本机当前配置下不应仅凭“FA3 理论更快”决定 backend；对 `window_pattern=L` 的 d24 Base 训练，PyTorch SDPA 是实测更优选择。

## 公平性控制

两组仅切换 NanoChat 统一 attention 接口的 backend，其他条件相同：

- 模型：d24，1,384,122,122 参数，width 1536，12 heads。
- 输入：真实 ClimbMix train loader，sequence length 2048。
- batch：device batch 2，8 ranks，total batch 524,288 token，gradient accumulation 16。
- 训练：BF16、`torch.compile(dynamic=False)`、forward、backward、分布式 MuonAdamW 通信和 optimizer step。
- 不包含：validation、CORE、采样、SwanLab/W&B、checkpoint。
- 每次 5 个 warmup step 后测量 10 个 step；运行顺序为 FA3 → SDPA → SDPA → FA3。
- 每轮使用相同 seed；step 时间取 8 个 rank 的最大值。

## 结果

| 顺序 | backend | 平均 step | 平均吞吐 | 中位 step | 峰值显存/rank |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | FA3 | 5.17843 s | 101,245 tok/s | 5.17695 s | 17,889.18 MiB |
| 2 | PyTorch SDPA | 4.98031 s | 105,272 tok/s | 4.97812 s | 15,942.59 MiB |
| 3 | PyTorch SDPA | 5.04917 s | 103,836 tok/s | 4.97621 s | 15,942.59 MiB |
| 4 | FA3 | 5.17126 s | 101,385 tok/s | 5.17187 s | 17,889.18 MiB |
| 合并 | FA3 | 5.17484 s | 101,315 tok/s | 5.17371 s | 17,889.18 MiB |
| 合并 | PyTorch SDPA | 5.01474 s | 104,549 tok/s | 4.97733 s | 15,942.59 MiB |

第二轮 SDPA 保留了 5.27608 秒和 5.40860 秒两个慢 step，没有为了美化结果删除。即便纳入这两个点，SDPA 的合并平均值仍优于 FA3；中位数结果也给出相同方向。

## 对完整 Base 的量级影响

若只按 11,136 个训练 step 外推，不计评测、采样和 checkpoint：

- FA3：约 16.01 小时。
- PyTorch SDPA：约 15.51 小时。
- 理论节省：约 29.7 分钟。

正式 Base 的 FA3 实测约 5.17 秒/step，和本次 FA3 对照一致，因此这次短测能够代表当前正式训练的稳态速度。

## 为什么可能出现 SDPA 更快

当前 FA3 兼容代码把旧版外部 CUDA op 包在 `@torch.compiler.disable` 中，以绕过 Torch 2.6 FakeTensor/data-pointer 问题。这会在每层 attention 处形成编译边界。PyTorch SDPA 则可以留在 `torch.compile` 图内，并由 CUDA SDPA dispatcher 选择 fused kernel。结合本次速度和显存结果，编译边界开销很可能抵消了 FA3 kernel 本身的优势。

这是基于源码与测量结果的解释，不代表 FA3 在其他 GPU、Torch/kernels 版本、序列长度或 sliding-window 配置中都更慢。

## 启动开销

编译 warmup 未计入稳态结果。第一轮冷启动中：

- FA3 首 step：10.22 秒。
- SDPA 首 step：48.77 秒。

长训练中该差异可以忽略；对于只有少量 step 的临时实验，FA3 的启动更快可能反而更重要。

## 证据位置

- 轻量汇总：`config/attention-backend-benchmark.json`
- 测速脚本：`scripts/attention_backend_benchmark.py`
- CPFS 原始 JSON/log：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/attention-benchmark/`
- 四份原始 JSON：`round1-fa3.json`、`round1-sdpa.json`、`round2-sdpa.json`、`round2-fa3.json`

## 后续建议

- 当前 d24、全上下文训练优先使用 PyTorch SDPA。
- 若升级 Torch、`kernels` 或移除 FA3 的 graph break，重新运行同一脚本，不沿用本次结论。
- 若将 `window_pattern` 改为包含 `S` 的 sliding window，单独重测；当前结论只覆盖 `L`。
