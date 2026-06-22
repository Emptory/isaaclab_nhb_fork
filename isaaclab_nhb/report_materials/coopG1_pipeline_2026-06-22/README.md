# CoopG1 阶段式强化学习报告材料包

生成日期：2026-06-22。材料覆盖 `coopG1S0`、`coopG1S1`、`coopG1S2` 的实验日志、配置快照、当前代码、工作区差异和已有图片。

## 建议阅读顺序

1. `docs/00_material_inventory.md`：原始路径、数据规模和材料总清单。
2. `docs/09_report_draft.md`：可直接继续修改的中文报告初稿。
3. `docs/01_report_outline.md`：完整报告结构和各章材料来源。
4. `docs/02_s1_tuning_timeline.md`：S1 全部实验的逐次调参记录。
5. `metrics/key_run_comparison.csv`：关键里程碑实验对照。
6. `docs/03_method_and_code_changes.md`：任务和代码修改过程。
7. `docs/04_experiment_findings.md`：现有结果能支持的结论及限制。
8. `docs/05_s2_residual_design.md`：S2 残差策略设计与当前完成度。
9. `docs/06_figure_index.md`：图片用途、图注建议和来源说明。
10. `docs/07_reproduction_commands.md`：训练、播放和材料重建命令。
11. `docs/08_data_dictionary.md`：CSV 字段和统计口径。

## 目录说明

- `configs/`：每次运行保存的 `env.yaml` 和 `agent.yaml`，这是精确参数的首要依据。
- `metrics/`：实验、checkpoint、观测项、奖励项和 TensorBoard 最终指标汇总。
- `metrics/timeseries/`：每个运行的降采样 TensorBoard 曲线数据，可用于重新制图。
- `hydra_outputs/`：204 次程序启动的 Hydra 配置快照，包含未形成正式训练目录的调试尝试。
- `figures/`：报告架构图、训练曲线和原始截图。
- `source_snapshots/`：当前 S0/S1/S2、RSL-RL 残差模块及 SteadyTray 参考代码快照。
- `metadata/`：Git 状态、提交记录、未提交代码差异和材料清单。
- `scripts/build_materials.py`：材料包的可重复生成脚本。

## 数据范围与注意事项

- 共索引 38 个运行目录、770 个 checkpoint 和 204 次 Hydra 启动；模型文件只建立索引，没有复制到材料包。
- `2026-06-22_11-26-35_s1_base_velocity_precise` 在采集材料时仍属于短程继续训练，结论只能视为阶段性结果。
- 不同实验修改过奖励权重，`Train/mean_reward` 不能直接跨运行比较；速度误差、任务成功率和视频表现更有可比性。
- 部分目录名用于记录调参意图。时间线中“推断目的”不是配置事实，精确参数必须回查 `configs/`。
