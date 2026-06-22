# 材料清单

## 原始材料位置

| 内容 | 原始路径 | 本材料包中的位置 |
|---|---|---|
| S0/S1/S2 训练日志 | `/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/` | `metrics/`、`configs/` |
| Hydra 启动快照 | `/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/outputs/` | `hydra_outputs/`、`metrics/hydra_output_index.csv` |
| 环境与训练代码 | `/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/` | `source_snapshots/isaaclab_nhb/` |
| RSL-RL 网络代码 | `/home/zhaowenhao/rsl_rl/` | `source_snapshots/rsl_rl/` |
| SteadyTray 参考实现 | `/home/zhaowenhao/SteadyTray/` | `references/SteadyTray/` |
| 用户原始图片 | `/home/zhaowenhao/图片/` | `figures/source_images/` |
| 生成的残差架构图 | `/home/zhaowenhao/.codex/generated_images/...` | `figures/architecture/` |

## 实验数据规模

| 阶段 | 运行目录 | Checkpoint | 原始模型体积 |
|---|---:|---:|---:|
| coopG1S0 | 4 | 83 | 0.797 GiB |
| coopG1S1 | 32 | 638 | 6.069 GiB |
| coopG1S2 | 2 | 49 | 0.458 GiB |
| 合计 | 38 | 770 | 7.324 GiB |

模型文件没有复制，`metrics/checkpoint_index.csv` 保留了每个文件的绝对路径、step 和大小。材料包还保存 204 次 Hydra 启动快照，其中包括没有形成正式 TensorBoard 运行的调试尝试。

## 可直接写入报告的文档

- `docs/09_report_draft.md`：中文初稿。
- `docs/01_report_outline.md`：章节结构。
- `docs/03_method_and_code_changes.md`：方法和实现过程。
- `docs/04_experiment_findings.md`：结果解释。
- `docs/05_s2_residual_design.md`：残差模块方法设计。
- `docs/10_missing_evidence_checklist.md`：正式定稿前待补实验。

## 调参记录

- `docs/02_s1_tuning_timeline.md`：逐运行可读时间线。
- `metrics/key_run_comparison.csv`：6 个关键里程碑。
- `metrics/run_summary.csv`：全部 38 个运行的统一表格。
- `metrics/reward_terms.csv`：每个运行的奖励函数及权重。
- `metrics/observation_terms.csv`：每个运行的观测项。
- `configs/`：原始环境与 agent 配置快照。

## 图片材料

- 1 张自有 S1/S2 残差架构图。
- 4 张自动生成的训练结果图。
- 24 张用户原始截图/参考图。
- 1 张原始图片 contact sheet，便于快速挑选。

详细用途和引用风险见 `docs/06_figure_index.md`。

