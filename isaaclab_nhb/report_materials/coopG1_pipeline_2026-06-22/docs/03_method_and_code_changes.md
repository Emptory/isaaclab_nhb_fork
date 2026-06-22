# 方法与代码修改过程

## 阶段划分

### S0：基础运动策略

S0 提供 29 DoF G1 的基础速度跟踪能力。其策略采用历史观测和普通 Actor-Critic MLP，为后续 S1、S2 提供运动先验。现有日志显示 S0 主要围绕步态、躯干角速度、骨盆姿态和足部朝向进行调整。

### S1：携物姿态下的正向行走

S1 继承 S0 环境，并将任务限定为正向行走：`lin_vel_x=(0.25, 0.4)`，横向速度和偏航角速度均为 0。奖励逐步加入或增强手部携物姿态、步态、骨盆/躯干稳定、足部朝向等目标。

S1 Policy 后续显式定义：

- `base_lin_vel`：机体坐标系线速度，噪声范围 `[-0.03, 0.03]`。
- `base_ang_vel`：机体坐标系角速度，噪声范围 `[-0.03, 0.03]`。

其中 `base_lin_vel` 是相对旧 S1 的新增项；`base_ang_vel` 在 S0/旧 S1 观测中已经存在，当前 S1 只是显式重申并统一噪声配置。实验对比应把变化表述为“新增线速度观测”，不能写成旧策略完全没有角速度观测。

加入后日志中的 Actor 输入维度为 530，动作输出为 29。网络仍由任务对应的普通 `ActorCritic` 配置自动创建，因此不会进入 S2 的残差网络。

为提高精确速度跟踪，当前 S1 在原有宽核速度奖励之外增加：

```text
track_lin_vel_xy_fine: weight=3.0, std=0.10
```

宽核用于保证学习初期的奖励覆盖，窄核用于在接近目标速度后继续区分误差。

### S2：固定负载残差策略

S2 继承 S1 的命令、基础观测、奖励和终止条件，增加固定负载、末端目标命令以及负载相对位姿/速度观测。任务注册提供普通配置和残差配置两个入口；残差配置选择 `ActorCriticResidual`，S1 仍选择 `ActorCritic`。

残差模块完成的代码机制包括：

- 从环境变量 `COOP_G1_S1_CHECKPOINT` 或默认路径加载 S1 checkpoint。
- 创建与 S1 一致的基础 Actor，并冻结其参数和归一化器。
- 残差 Actor 使用 `256-128-64` MLP。
- 将残差输出经过 `tanh` 和 `residual_scale=0.1` 限幅。
- 最后一层零初始化，使训练起点与 S1 行为一致。
- 最终动作：`a = a_base + 0.1 * tanh(delta_a)`。

## 其他工程修改

- RSL-RL 模块导出并注册 `ActorCriticResidual`，Runner 可根据任务配置动态实例化对应网络。
- checkpoint 加载默认映射到 Runner 当前设备，解决设置 `CUDA_VISIBLE_DEVICES=7` 后逻辑设备只有 `cuda:0`、而 checkpoint 记录物理 `cuda:7` 的反序列化错误。
- S1 播放初始偏航由 `1.57` 调整为 `0`，避免世界坐标显示方向与任务正向定义不一致。

## 可追溯证据

- 当前实现：`source_snapshots/isaaclab_nhb/` 和 `source_snapshots/rsl_rl/`。
- 尚未提交的精确修改：`metadata/isaaclab_nhb_worktree.diff`、`metadata/rsl_rl_worktree.diff`。
- 每次训练实际使用的配置：`configs/<stage>/<run>/`。
- 完整时间线：`docs/02_s1_tuning_timeline.md`。
