# Smoke Attempt 1

- 时间：2026-08-06T14:19:43+00:00
- 运行 commit：`a5c4f64`
- 结果：失败，训练 step 尚未开始
- GPU 计算：未开始
- checkpoint：未生成
- SwanLab 正式 run：未生成

## 根因 1：launcher 使用了错误的 Python

实验 venv 通过 `--system-site-packages` 复用镜像的 PyTorch，但 venv 内没有独立的 `torchrun` console script。Shell 回退到 `/usr/local/bin/torchrun`，其 shebang 启动系统 Python；8 个 worker 因此看不到只安装在实验 venv 中的 W&B/SwanLab。

首个可见错误：

`ModuleNotFoundError: No module named 'wandb'`

## 根因 2：生成脚本的续行被破坏

初版脚本使用反斜杠续行。生成 Markdown/脚本资产时，续行被折叠为同一行的字面量 `+` 参数。`bash -n` 只能验证语法合法，未发现参数语义错误。

## 修复

1. 所有多卡入口改为 `python -m torch.distributed.run`。
2. 所有训练命令改用 Bash 数组，不再使用反斜杠续行。
3. 修复后先展开并执行 CLI 导入检查，再重新执行完整 0→50→100 冒烟。
4. Attempt 1 的任何文件都不作为成功实验产物。

完整日志：

`/data/cache/nanochat/experiments/l20-d24-swanlab-20260806/smoke-supervisor.log`

- bytes：7036
- SHA256：`72b6be1ae5d6e9a0e30ee98f816016b230425093b3dccbb4652a1b840c679302`
