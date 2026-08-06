# d24 Base 容量与速度探针

## 结论

Attempt 2 通过。8×L20、BF16、FA3、d24、正式 total batch 的 3-step 训练与完整 checkpoint 保存成功。

## Attempt 1

- 失败发生在首个训练 step 前。
- Torch 2.6 Dynamo 追踪进 FA3 自定义 CUDA op，FakeTensor 无法访问 data pointer。
- supervisor log SHA256：`abdb2129e01c7c82bfcca285c6a0d3782106da61b6adb2dcb1f36de8b4f7d22d`

## 修复

- `kernels 0.16.0` 与当前源码 API 不兼容，按锁文件降为 `0.11.7`。
- FA3 训练调用以 `torch.compiler.disable` 保持 opaque。
- Torch 2.6 的 tuple 返回值归一化为 output Tensor。
- kernel revision 固定为 `9542c462013476380ce4b395b9ddc0e8118161ee`。
- 20 项 FA3/SDPA GPU 测试通过。

## Attempt 2 配置

- 参数量：1,384,122,122
- depth/width/heads：24 / 1536 / 12
- sequence length：2048
- device batch：2
- total batch：524,288 tokens
- gradient accumulation：16
- attention：FA3
- SwanLab run ID：`0spgv31a`

## 性能

| step | loss | dt | token/s |
| ---: | ---: | ---: | ---: |
| 0 | 10.398221 | 26.113 s | 20,078 |
| 1 | 10.380246 | 5.165 s | 101,508 |
| 2 | 10.350516 | 5.174 s | 101,332 |

- 峰值 allocated memory：17,889 MiB/rank。
- 采样到的 device memory 峰值约 24 GiB，46 GiB L20 仍有充足余量。
- checkpoint：1 个约 3.94 GiB 的模型、8 个各约 684 MiB 的 optimizer shard、1 个 metadata。
- CPFS 当前使用约 33 GiB / 3.6 TiB。

## 正式训练估算

- ratio=8 对应 21,120 step、11,072,962,560 token。
- 纯训练按 5.17 秒/step 约 30.3 小时。
- 加上 validation、CORE、sample 与 checkpoint，预计 31–33 小时。

## 产物哈希

- Attempt 2 supervisor log：`6014ca951d617ed55434452b9f08cf36ad72e8841367a26b4be0a8297c02be32`
- metadata：`ce0f1b24abf069a2414b5d6f4da9d0494ce3d64fc94dab6fafdb85cce79290bf`
- model：`742bbcee137c0b86d1fe8020dcb6e5959c3f1f47aa098310ae58dcc308acfa5f`
- SwanLab：`14003cae6e785d451ca78eb51aa61ac97f0098e2dc53ed4849d92f3b412a1a57`

原始日志、checkpoint 和 SwanLab run 保存在 CPFS，GitHub 保存摘要和哈希。
