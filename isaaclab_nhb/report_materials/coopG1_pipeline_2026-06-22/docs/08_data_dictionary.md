# 数据字典

## `metrics/run_summary.csv`

每行对应一个运行目录。重要字段：

- `experiment`、`run`、`run_path`：阶段、运行名和原始路径。
- `num_envs`、`max_iterations_cfg`、`seed`：配置中的训练规模。
- `actual_last_iteration`：TensorBoard 最后记录的全局 step。
- `latest_checkpoint_step`：目录中编号最大的 checkpoint。
- `resume`、`load_run`、`load_checkpoint`：初始化来源。
- `policy_terms`、`policy_history_length`：Policy 观测项和历史长度。
- `policy_has_base_lin_vel`、`policy_has_base_ang_vel`：是否包含机体速度观测。
- `track_vel_weight/std`、`track_vel_fine_weight/std`：速度奖励参数。
- `final_*`：对应 TensorBoard tag 的最后记录值。

`actual_last_iteration` 可能大于 checkpoint 编号，因为 resume 会沿用全局 step，或事件文件继续记录但保存间隔尚未到达。

## 其他 CSV

- `checkpoint_index.csv`：770 个模型的路径、step 和文件大小；不包含模型本体。
- `hydra_output_index.csv`：204 次 Hydra 启动的日期、时间和关键 agent 配置，包括未产生训练日志的调试启动。
- `observation_terms.csv`：从每次 `env.yaml` 提取的观测组与观测项。
- `reward_terms.csv`：每个奖励项的函数、权重和参数。
- `final_scalar_metrics.csv`：每个运行、每个 TensorBoard tag 的最后值与 step。
- `key_run_comparison.csv`：人工筛选的关键实验里程碑。
- `metrics/timeseries/<stage>/<run>.csv`：降采样后的完整曲线，字段为 `tag, step, wall_time, value`。

## 统计注意事项

- `final` 表示日志最后一个采样点，不一定是训练中的最优 checkpoint。
- 不同运行的命令范围、奖励项和权重不同，不能仅按 mean reward 排名。
- 只有 `num_envs=1`、`max_iterations=1` 的目录通常是播放或测试记录，应从训练对比中排除。
- `configs/` 保存运行时配置；`source_snapshots/` 保存 2026-06-22 当前代码，两者可能不一致。
- `hydra_outputs/` 更接近每次程序启动快照，但命令使用 argparse 解析时 `overrides.yaml` 可能为空；最终训练参数仍以运行目录配置为主。
- UTF-8 CSV 含 BOM，Excel 可直接打开；脚本读取建议使用 `encoding="utf-8-sig"`。
