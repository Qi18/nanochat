# SFT 独立全量评估与 Base 对比

## 结论

SFT checkpoint `model_000932.pt` 已完成两套独立全量评估：

1. Chat Eval 衡量指令模型在 ARC、MMLU、GSM8K、HumanEval 上的能力，完整 ChatCORE 为 `0.164724`；
2. Base BPB/CORE 同协议评估衡量 SFT 前后的基础语言建模和 ICL 能力，SFT 后 val BPB 增加 20.29%，CORE 相对下降 12.62%。

两套协议目的不同，不能直接比较各自的绝对分数。

## 全量 Chat Eval

- 时间：2026-08-07 12:36:08 UTC 至 12:44:14 UTC；
- 硬件：8×NVIDIA L20；
- dtype：BF16；
- attention：PyTorch SDPA；
- 分类 batch：8；
- 生成：temperature 0、1 sample、最多 512 tokens；
- `max_problems`：未设置，评测全部样本。

| 任务 | 正确/总数 | Accuracy | 训练内抽样 | 差异 |
| --- | ---: | ---: | ---: | ---: |
| ARC-Easy | 1240/2376 | 52.19% | 53.60%（500） | -1.41pp |
| ARC-Challenge | 436/1172 | 37.20% | 37.40%（500） | -0.20pp |
| MMLU | 4841/14042 | 34.48% | 33.80%（500） | +0.68pp |
| GSM8K | 42/1319 | 3.18% | 4.17%（24） | -0.99pp |
| HumanEval | 23/164 | 14.02% | 16.67%（24） | -2.65pp |
| ChatCORE | - | 0.164724 | 0.174470 | -0.009746 |

训练内评估适合观察趋势，但生成任务只有 24 条，方差较大；最终结论采用本次全量结果。

SwanLab：[SFT full Chat Eval](https://swanlab.cn/@richliu0153/nanochat-lab/runs/i239hpv4)。

## Base 同协议 BPB/CORE

两次评估均使用：8×L20、BF16、FA3、每个 split 20,971,520 tokens、完整 22 项 CORE、`max_per_task=-1`。

| 指标 | Base step 11136 | SFT step 932 | 变化 |
| --- | ---: | ---: | ---: |
| train BPB | 0.714492 | 0.858575 | +20.17% |
| val BPB | 0.712537 | 0.857091 | +20.29% |
| CORE | 0.259960 | 0.227166 | -12.62% |

22 项 CORE 中：

- 6 项提升：CommonsenseQA、BigBench CS Algorithms、BigBench Operators、CoQA、BoolQ、Language Identification；
- 1 项持平：AGI Eval LSAT AR；
- 15 项下降。

最大 raw accuracy 提升：

- BigBench CS Algorithms：`0.393182 → 0.454545`；
- CommonsenseQA：`0.264537 → 0.295659`；
- BoolQ：`0.526911 → 0.548930`。

最大 raw accuracy 下降：

- BigBench QA Wikidata：`0.462428 → 0.364647`；
- HellaSwag：`0.561243 → 0.478789`；
- COPA：`0.680000 → 0.610000`。

SwanLab：[SFT Base-protocol full Eval](https://swanlab.cn/@richliu0153/nanochat-lab/runs/ln0ha6dv)。

## 判断

本轮 SFT 没有异常提前结束，Chat validation BPB 也持续下降；但它优化的是监督对话分布。相同 Base 数据分布上的 BPB 和 CORE 同时退化，说明存在可测量的基础能力遗忘。

进入 RL 前建议先做一轮小规模 SFT 配方对照，优先比较：

1. 降低 SFT 学习率或缩短训练 token；
2. 在 SFT mixture 中加入一定比例的 Base replay 数据；
3. 每 100 step 同时记录 Chat validation BPB 与固定子集 Base BPB/CORE，选择 Pareto checkpoint，而不是只选 SFT validation 最低点。

## 产物与校验

轻量结果：

- `../config/sft-eval-results.json`；
- `sft-chat-eval.csv`；
- `base-sft-core-comparison.csv`。

CPFS 原始结果：

- Chat Eval log：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/sft-eval.log`；
- Chat Eval log SHA256：`aa3b72c3461f4475e022e7f17efdf7847407f4cac67b2e1b971e61b6e4706807`；
- Base-protocol log：`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/sft-base-protocol-eval.log`；
- Base-protocol log SHA256：`986d28cd8cf4e6396a0cfa1e6860d928f5c64183c94be2fd710a9b8eb06163b8`；
- Base-protocol CSV：`/data/cache/nanochat/base_eval/sft_model_000932.csv`；
- Base-protocol CSV SHA256：`37a13f2dd204e0bfa5abd7870474f79c1bbbc75cf0efdf6de755ccf7856dbcea`。
