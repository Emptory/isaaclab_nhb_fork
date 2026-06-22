# 图片索引与使用建议

## 可直接用于方法章节

| 文件 | 建议位置 | 建议图注 |
|---|---|---|
| `figures/architecture/s1_s2_residual_policy_architecture.png` | 方法总览 | 阶段式 S1 基础策略与 S2 残差策略架构。S1 checkpoint 在 S2 中冻结，残差网络根据末端目标和负载状态修正基础动作。 |
| `figures/source_images/img-001.jpg` | 相关工作 | 冻结基础策略并通过动作残差或 FiLM adapter 适配新任务的参考架构。 |
| `figures/source_images/img-002.jpg` | 相关工作 | privileged encoder 与 student encoder 的教师-学生适配参考架构。 |
| `figures/source_images/figure2.png` | 相关工作 | 末端目标跟踪、补偿控制和 PPO Actor-Critic 联合训练参考框架。 |
| `figures/source_images/fig01.jpg` | 相关工作 | 普通 Actor-Critic 与可微 MPC 结合架构参考图。 |
| `figures/source_images/figure1.png` | 相关工作 | 多智能体协作与 Lyapunov 约束优化参考图。 |

上述五张文献图只能在确认论文出处和引用许可后用于正式报告；当前材料包保留的是用户提供的参考图片，没有自动补全文献来源。

自有架构图表达的是目标方案。与当前代码相比有两点需在定稿时修正：上层 hand command 只应进入 residual observation，不进入冻结 S1 Actor；当前 `alpha` 是统一的标量 `0.1`，还不是逐关节缩放向量。

## 可直接用于实验章节

| 文件 | 用途 | 注意事项 |
|---|---|---|
| `figures/training/s1_selected_training_curves.png` | 对比关键 S1 运行的速度误差与宽核速度奖励 | resume 运行沿用全局 step，横轴起点可能不是 0。 |
| `figures/training/latest_s1_dashboard.png` | 展示最新有效 S1 运行的误差、奖励、总回报和噪声 | 6 月 22 日运行是阶段性数据。 |
| `figures/training/s1_reward_contributions_iter9999.png` | 展示 6 月 21 日模型的主要正负奖励贡献 | 是带权奖励贡献，不是未加权物理误差。 |
| `figures/training/s1_final_velocity_error_all_runs.png` | 附录展示所有 S1 运行最终速度误差 | 小规模测试和不同命令范围也在图中，不适合直接排名。 |

## 原始截图

`figures/source_images/1.png` 至 `13.png` 以及所有 `Snipaste_*.png` 均从 `/home/zhaowenhao/图片` 原样复制。建议人工逐张确认画面内容后再命名，例如：

- `1.png`、`2.png`、`3.png`：机器人双手携箱效果，可选作任务场景或定性结果图。
- `Snipaste_2026-06-07_10-30-34.png`：正视角携箱效果，适合与侧视图组成双栏图。
- `4.png`、`5.png`、`7.png`、`10.png` 至 `13.png` 及其余 `Snipaste`：带末端/方向 marker 的步态调试画面，可用于调参过程或失败案例。
- `6.png`：Isaac Sim 编辑器全屏截图，信息较杂，建议只用于开发过程附录。
- `figures/source_images_contact_sheet.png`：全部 24 张原图的缩略图索引，便于快速挑选，不建议直接放入正文。

```text
S1_forward_walk_command_035.png
S1_hand_pose_failure_case.png
S1_pelvis_oscillation_comparison.png
S2_fixed_payload_scene.png
```

正式报告优先使用无遮挡、同一视角、同一命令和相同帧尺寸的对比截图；动态效果应另录视频并从同一时间点截帧。
