# d24 Base 独立评测

## 结论

最终 `model_011136.pt` 已完成独立 `sample,bpb,core` 评测。评测使用 8×L20、BF16、完整 CORE 数据集，以及 train/val 各 20,971,520 token；未出现 OOM、NaN、NCCL 或 rank 异常。

- train BPB：0.714492
- validation BPB：0.712537
- 完整 CORE：0.259960

训练末尾的 CORE 0.27082 使用每任务最多 500 条数据，本次 `max_per_task=-1` 不做截断，因此以本次 0.259960 作为正式 Base Eval 结果。

## 配置

- checkpoint：`/data/cache/nanochat/base_checkpoints/d24-l20-swanlab-20260806/model_011136.pt`
- ranks：8
- device batch：2
- sequence length：2048
- split tokens：20,971,520
- CORE max per task：-1（完整数据集）
- modes：sample、bpb、core
- 用时：约 12 分 53 秒

## CORE 逐任务结果

| Task | Accuracy | Centered |
| --- | ---: | ---: |
| hellaswag_zeroshot | 0.557459 | 0.409945 |
| jeopardy | 0.102031 | 0.102031 |
| bigbench_qa_wikidata | 0.462428 | 0.462428 |
| arc_easy | 0.688552 | 0.584736 |
| arc_challenge | 0.385666 | 0.180887 |
| copa | 0.680000 | 0.360000 |
| commonsense_qa | 0.264537 | 0.080672 |
| piqa | 0.745375 | 0.490751 |
| openbook_qa | 0.400000 | 0.200000 |
| lambada_openai | 0.442461 | 0.442461 |
| hellaswag | 0.561243 | 0.414990 |
| winograd | 0.688645 | 0.377289 |
| winogrande | 0.557222 | 0.114444 |
| bigbench_dyck_languages | 0.147000 | 0.147000 |
| agi_eval_lsat_ar | 0.260870 | 0.076087 |
| bigbench_cs_algorithms | 0.393182 | 0.393182 |
| bigbench_operators | 0.185714 | 0.185714 |
| bigbench_repeat_copy_logic | 0.031250 | 0.031250 |
| squad | 0.424503 | 0.424503 |
| coqa | 0.313667 | 0.313667 |
| boolq | 0.526911 | -0.244970 |
| bigbench_language_identification | 0.247400 | 0.172057 |
| **CORE** |  | **0.259960** |

## 采样判断

固定事实提示可以给出 Paris、Au、cold 等正确短答案，但无条件长文本仍有事实混淆、重复与伪引用。Base 模型已经学到语言结构和部分知识，尚未稳定形成指令跟随能力；这与下一阶段进行 SFT 的目标一致。

## SwanLab

- run ID：`n8d4bcb1`
- run：<https://swanlab.cn/@richliu0153/nanochat-lab/runs/n8d4bcb1>
- 上传记录：107

## 证据

- 轻量 CSV：[base-eval.csv](base-eval.csv)
- 配置与机器可读结果：`config/base-eval-results.json`
- 原始日志：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/base-eval.log`
- 原始 CSV：`/data/cache/nanochat/base_eval/base_model_011136.csv`
- 原始日志 SHA256：`931e10c6fafde3043b6acf6679635afa07b16b2898142e66e1098b9b680c66e9`
- 原始 CSV SHA256：`84d1aff9e5a93e73c49d49ebbe2fce17dc0b4a59ac74f228ab97dfc772fef510`
