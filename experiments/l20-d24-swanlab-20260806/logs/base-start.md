# d24 Base 启动记录

## 状态

已完成。最终结果见 [base-final.md](base-final.md)。

- 启动 commit：`3126420e9662c87e299b6cf4c65398b5b8936696`
- 启动时间：2026-08-06 14:54 UTC
- SwanLab offline run ID：`thapegmb`
- 进程：8 ranks
- dtype：bfloat16
- attention：FA3
- kernel revision：`9542c462013476380ce4b395b9ddc0e8118161ee`

## 模型与 horizon

- 总参数：1,384,122,122
- scaling params：729,810,624（transformer matrices + lm_head）
- target ratio：8
- iterations：11,136
- 实际训练 token：5,838,471,168
- total batch：524,288 token
- gradient accumulation：16

## 启动证据

- Step 0 validation bpb：3.160318
- Step 0 loss：10.398221
- Step 1 loss：10.380246
- Step 2 loss：10.350516
- 稳态约 5.18–5.20 秒/step。
- 采样显存约 22–24 GiB/rank。
- 启动阶段未出现 NaN、Inf、OOM 或 rank 退出。

原始日志：CPFS `base-formal-supervisor.log` 与 `base-train.log`。SwanLab run 已同步到云端。
