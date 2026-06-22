# S2 残差策略设计

## 当前架构

S2 使用“冻结 S1 基础策略 + 可训练残差策略”：

```text
a_base = pi_S1(o_base)
delta_a = tanh(pi_res(o_res))
a_final = a_base + alpha * delta_a, alpha = 0.1
```

基础策略 MLP 与 S1 checkpoint 匹配，残差 Actor 为 `256-128-64-29`；Critic 为 `512-256-128-1`。残差输出层零初始化，因此初始时 `delta_a=0`，不会一开始破坏 S1 已学会的行走。

## 当前 residual observation

代码中的 `residual_policy` 目前包含完整 S1 Policy 观测，并追加：

- 上层给出的左右手末端目标 `hand_target`。
- 负载相对躯干的位置与四元数。
- 负载相对躯干的线速度与角速度。
- 负载坐标系中的重力方向。
- 左右手在负载坐标系下的位置。

该组合能表达机器人本体状态、基础运动命令、负载状态和末端几何误差，适合作为第一版残差网络输入。

## 在目标轨迹和目标力之外还应补充的输入

上层只给目标轨迹和目标力还不够。残差模块需要看到“目标与实际之间的误差”和接触状态，建议补充：

- 末端实际位置、姿态、线速度、角速度，以及对应目标误差。
- 左右手实际接触力/力矩、目标力/力矩和误差。
- 接触标志、滑移速度、法向方向和左右手受力分配。
- S1 输出的 `a_base` 或上一时刻最终动作，帮助残差判断正在修正什么。
- 轨迹相位、剩余时间或短时域目标点，避免只凭单帧目标产生迟滞。
- 负载质量、质心偏置等 privileged 参数只给 Critic；部署 Actor 使用可测量量或估计 latent。

## 推荐输入分组

```text
o_res = [
  o_S1,
  a_base,
  hand_target,
  hand_pose_error,
  hand_twist_error,
  wrench_target,
  wrench_error,
  contact_state,
  payload_pose_twist,
  trajectory_phase
]
```

## 当前完成度与训练前缺口

当前代码已经搭好网络选择、S1 checkpoint 冻结、残差限幅和负载观测框架，但尚未完整实现上层目标力、真实接触 wrench、`a_base` 回灌以及轨迹相位。正式训练 S2 前，应先确认这些量的坐标系、量纲、归一化和传感器来源，再做短程 smoke test。

建议先做三组消融：S1 基础策略、S1+仅轨迹残差、S1+轨迹与力反馈残差。评测使用相同负载扰动和命令序列。

