# 本次运行计划

## 基线

- 分支：`experiment/l20-d24-swanlab-20260806`
- 基线 commit：以 `config/provenance.json` 中的 `git_commit` 为准
- GPU：8×NVIDIA L20
- dtype：BF16
- Attention：`window-pattern=L`
- SwanLab project：`nanochat-lab`
- SwanLab group：`l20-d24-swanlab-20260806`

## 执行顺序

1. `scripts/00_preflight.sh`
2. 提交 `config/provenance.json`
3. `scripts/01_prepare_data.sh`
4. 提交 `config/data_manifest.json` 和 tokenizer 摘要
5. `scripts/02_smoke.sh`
6. 核对 step 50 → step 100 恢复证据并提交摘要
7. `scripts/03_base_train.sh`
8. Base Eval、SFT、RL 和统一评测
9. 汇总源码笔记、博客文章与 `FINAL_REPORT.md`

## 训练门禁

- 仓库必须干净，实际 commit 必须已推送。
- 数据 shard 与 tokenizer 清单必须已记录。
- loss/BPB 不得为 NaN/Inf。
- checkpoint 必须能够加载并继续训练。
- SwanLab 在线失败时允许离线记录，但本地 run 目录必须存在。
