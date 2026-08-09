# SFT 探针与启动前修复

## 数据预检

SFT 数据已通过 `hf-mirror.com` 下载并缓存到 CPFS：

- SmolTalk：train 460,341 行，test 24,229 行；
- MMLU：auxiliary_train 99,842 行，test 取 5,200 行；
- GSM8K：train 7,473 行，test 取 420 行；
- 正式训练混合：789,759 行，MMLU 重复 3 次，GSM8K 重复 4 次；
- 抽样监督 token 均有效，未发现全 padding target。

## 启动前修复

第一次 3-step 探针暴露两个可复现问题：

1. `tasks/common.py` 将 Hugging Face 地址硬编码为 `huggingface.co`，导致 `HF_ENDPOINT` 不生效。现已让下载逻辑读取 `HF_ENDPOINT`，并重写 API 返回的 shard URL。
2. `scripts/chat_sft.py` 的 `--num-iterations` 表示 optimizer step，但数据生成器按 micro-batch 计数。在梯度累积为 16 时，3-step 探针只供应 3 个 micro-batch，造成首步进度 566.67%、学习率倍率为 -9.33。现已按 `num_iterations * grad_accum_steps` 供应 micro-batch，并将进度限制在 `[0, 1]`。

`tests/test_tasks.py` 修复后 7 项通过。无效探针没有删除，已隔离到：

- checkpoint：`/data/cache/nanochat/chatsft_checkpoints_quarantine/d24-l20-swanlab-20260806-invalid-num-iterations-probe`
- log：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/sft-probe-invalid-num-iterations.log`
- log SHA256：`b0f99509719267c2a097977f35f0f54ff9516829a2943505599ad87633dbd5f3`

## 有效 3-step 探针

修复后的探针结果：

- 进度：33.33% → 66.67% → 100.00%；
- LR multiplier：1.00 → 0.67 → 0.00；
- step 2/3：约 4.98 秒/step、105K token/s；
- validation BPB：0.4394 → 0.4208；
- 峰值显存：15,943.21 MiB/rank；
- checkpoint：model、metadata、8 份 optimizer shard 均完整；
- SwanLab：[5afbyxmz](https://swanlab.cn/@richliu0153/nanochat-lab/runs/5afbyxmz)；
- log SHA256：`0390553e89c50fe9f73a5458f9a7f1d9d82169c5664bfb6d712828943bfe5cf5`。

有效探针 checkpoint 保存在：

`/data/cache/nanochat/chatsft_checkpoints_probe/d24-l20-swanlab-20260806-step3`
