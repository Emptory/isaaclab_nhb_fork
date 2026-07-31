# S2 双手力跟踪实现说明

更新日期：2026-07-26

## 1. 已实现范围

S2 已接入 ICRA 2024 *Learning Force Control for Legged Manipulation*
的核心三维虚拟弹簧结构。参考源码为
`Improbable-AI/learning-compliance`，本地核对 commit：
`c760e1d74ad165d3c069d4f57ab5d066f6a41eb6`。

当前实现：

- 左、右掌面各有独立的三维移动 spring anchor；
- 每个 0.005 s physics substep（200 Hz）重新读取掌面位置/速度并计算外力；
- 虚拟外力施加在真实掌面 anchor，而不是 hand-link COM；
- Actor 读取目标力和 force-control 轴，不读取真实力；
- 一个监督 estimator 从可部署的 tracking/proprioception history 预测两手实际力；
- Critic 和 estimator loss 读取精确的 virtual-force privileged truth；
- force reward、clamp penalty、episode metrics、play CSV 和实时曲线已接入；
- checkpoint schema 升为 v5，且指纹包含弹簧轴、Kp/Kd、限幅、力符号和掌面 anchor；旧 S2 checkpoint 被明确拒绝。

当前只实现每手 3D force，共 6 个标量；未实现 contact moment tracking。作用点
不在 COM 时自然产生的 `r x F` 是施力点的物理结果，不是力矩跟踪。

## 2. 弹簧方程和更新频率

对每只手 \(i\)：

\[
F_{\mathrm{ext},i}
=K_p(x_{s,i}-x_{g,i})+K_d(v_{s,i}-v_{g,i})
\]

- \(x_s,v_s\)：reset-aligned CSV 轨迹给出的移动 spring anchor；
- \(x_g,v_g\)：当前掌面 anchor 的位置和速度；
- \(F_{\mathrm{ext}}\)：环境施加到机器人手上的外力；
- \(F_{\mathrm{cmd}}\)：独立的目标力，只用于 observation/reward，不直接代入弹簧。

默认使用论文数值 \(K_p=700\,\mathrm{N/m}\)、\(K_d=6\,\mathrm{Ns/m}\)，并按轴
限制到 20 N。公开 release 实现使用 \(0-v_g\)；这里按论文公式，并针对移动的
offline anchor 使用 \(v_s-v_g\)。当前 0.30 m/s 移动轨迹若直接套用
\(0-v_g\)，会额外产生 1.8 N 的阻尼偏置。

更新 hook 位于 S2 专用 `VirtualSpringJointPositionAction.apply_actions()`：

1. 写入 joint position target；
2. 重新计算 virtual spring；
3. `scene.write_data_to_sim()`；
4. PhysX step。

因此不是在 50 Hz policy step 只算一次后保持四个 substep。

## 3. 力的符号

MATLAB CSV 的力来自箱体动力学：

\[
m\dot v=F+mg
\]

并经 grasp matrix 分到左右手，所以 CSV 保存的是：

\[
F_{\mathrm{CSV}}=F_{\mathrm{hand\rightarrow payload}}
\]

而论文和仿真 spring 保存的是：

\[
F_{\mathrm{ext}}=F_{\mathrm{environment\rightarrow hand}}
\]

代码只做一次符号转换：

\[
F_{\mathrm{cmd}}=-F_{\mathrm{CSV}}
\]

当前 CSV 每手为 `[0, 0, +0.24525] N`，即每只手向上支撑箱体；机器人手应
感受到 `[0, 0, -0.24525] N` 的环境反作用。

play CSV 同时保存：

- `target_csv_hand_on_payload_force`：原始 CSV 符号；
- `target_force`：训练目标，environment-on-hand；
- `actual_force`：真正写给 PhysX 的 virtual spring force；
- `estimated_force`：Actor 内部 estimator 的输出。

## 4. 坐标和施力点

spring 计算时先将 position/velocity error 旋转到 reset-aligned dataset inertial
frame N，在 N 中应用 force-axis mask、增益和 clamp，再旋转到 world。

调用 Isaac Lab：

```python
robot.set_external_force_and_torque(
    forces=force_w,
    torques=zeros,
    positions=anchor_position_w,
    body_ids=hand_body_ids,
    is_global=True,
)
```

Isaac Sim 5.1 在 `is_global=True` 时，force、torque 和 position 都要求 world
坐标。不能把 world force 与 link-local anchor offset 混用。

## 5. 当前 hybrid 模式

论文使用 position/force 互斥模式。当前搬运任务采用逐轴 hybrid 扩展：

- x/y：position 和 linear-velocity tracking；
- z：virtual-force tracking；
- orientation/angular velocity：继续跟踪，因为当前没有 moment spring。

配置为：

```text
force_control_axes = (0, 0, 1)
position_control_axes = (1, 1, 0)
```

把 mask 改成 `(1,1,1)` 可切换到三维 direct-force 模式。force 轴的 position/
velocity reward 已关闭，避免同一 spring deflection 同时被位置奖励拉回零。
Actor 仍保留有方向的误差 history，供无力传感器 estimator 推断 virtual force。

## 6. Actor、estimator 和 Critic

部署观测：

- residual tracking/proprioception history：207D；
- target force + force-axis directions：12D；
- 当前 frozen-S1 arm action：14D，在 policy 内重算并附加。

control normalizer 只处理 `207 + 12 + 14 = 233D`。estimator 从其中的
207D residual history 预测：

\[
\hat F=[F_L^x,F_L^y,F_L^z,F_R^x,F_R^y,F_R^z]
\]

预测值在 normalizer 后附加，Residual Actor 输入为 `233 + 6 = 239D`。
真实 virtual force 不进入 Actor。

Critic 输入：

- 原 S2 kinematic critic history：850D；
- force context：12D；
- current exact virtual force：6D；
- 合计：868D。

estimator 使用 privileged current virtual force 的 Smooth-L1 loss；原始
TensorDict observation 随 RolloutStorage 保存，PPO 每个 replay epoch 用当前
estimator 权重重算预测值，不保存 stale estimate。

## 7. Reward

运动学 reward：

- position fine/wide 只作用于 x/y；
- linear velocity 只作用于 x/y；
- orientation、angular velocity 保持三维。

force reward 使用论文形式并逐手平均：

\[
r_F=\frac{1}{2}\sum_{i\in\{L,R\}}
\exp\left(-\frac{\|F_{\mathrm{ext},i}-F_{\mathrm{cmd},i}\|}{\sigma_F}\right)
\]

当前 CSV 只有 0.24525 N，因此尺度按任务缩小：

- fine：weight 5.0，\(\sigma_F=0.25\,N\)；
- wide：weight 1.0，\(\sigma_F=1.0\,N\)；
- clamp fraction：weight -0.10。

不能直接使用论文面向几十牛顿命令的 20 N reward scale，否则从 0 到
0.24525 N 的误差几乎没有区分度。

## 8. 尚未完成或不能宣称的内容

1. 当前 CSV 只有 z 轴恒定力，x/y 均为零。因此网络结构支持双手 3D force，
   但这份数据只能验证 z-force tracking，不能证明通用 3D 力控制。
2. 当前 actual force 是仿真已知 virtual spring truth，不是腕部 F/T sensor 或
   真实接触力。
3. 尚未实现论文的随机多档 force command、ramp/hold/ramp-down、约 20%
   spring-off free/compliance 样本和 stiffness/damping domain randomization。
   在宣称一般化 force control 前必须补齐这些训练分布。
4. \(K_p=700\) 时 0.24525 N 只对应约 0.35 mm spring deflection。结构与论文
   一致，但对 G1 的 action/PD/数值尺度可能过小；运行 smoke test 后应依据
   clamp fraction、force error 和 estimator error决定降低 stiffness或扩大
   force-command curriculum。
5. 没有 moment tracking，也没有真实 payload/contact dynamics。

本次只进行了源码修改、静态编译和小型 CPU policy 维度测试；没有启动训练或
Isaac Sim。
