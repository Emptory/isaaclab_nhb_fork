# S2 理论与静态代码审计

审计日期：2026-07-26

## 1. 当前结论

当前 S2 的源码链路在理论上包含两部分：

1. 冻结 S1 的 locomotion/carry policy 加 14 个上肢 residual；
2. x/y 运动学跟踪加 z 方向 paper-style virtual-force tracking。

已经静态确认：

- S1 observation 的 previous action 仍是 frozen-base action；
- S2 residual 的随机变量、PPO storage 和 log probability 都位于 latent
  Gaussian 空间；
- virtual spring 在每个 physics substep 更新；
- CSV hand-on-payload force 到 environment-on-hand force 只做一次负号转换；
- Actor 不读取 exact actual force；
- estimator 用可部署 history 预测 actual force；
- Critic 和 estimator supervision 才读取 privileged actual virtual force；
- force-control z 轴不再同时获得 position/linear-velocity reward；
- force application point、world frame 和左右手顺序明确；
- schema v5 指纹覆盖新 observation、estimator、动作，以及弹簧轴、Kp/Kd、限幅、力符号和掌面 anchor。

这仍然是静态理论审查，不代表已经学会。未运行训练或 Isaac Sim。

## 2. Policy 数据流与维度

### Frozen S1

- 输入：原 530D S1 observation（legacy task 为 515D）；
- previous action：上一步 frozen S1 action；
- 输出：29D `a_base`；
- actor 和 normalizer 全部冻结、eval。

### Deployable residual control

Residual history group 为 207D：

| 项目 | 维度 |
|---|---:|
| 双手当前 p/q/v/omega reference | 26 |
| p/R/v/omega directional error，5 帧 | 120 |
| 14 个上肢关节 q/qd | 28 |
| gravity、base velocities、velocity command | 12 |
| gait command | 7 |
| previous arm residual | 14 |
| 合计 | 207 |

Force context 为 12D：

- target environment-on-hand force：6D；
- 两手 force-control axis directions：6D。

policy 内部重算当前 frozen-S1 arm action 14D：

```text
deployable control = 207 + 12 + 14 = 233D
```

control normalizer 只处理这 233D。force estimator 从 normalized 207D history
预测两手 3D actual virtual force，共 6D；预测值附加到 Actor：

```text
Residual Actor = 233 + estimated force 6 = 239D
```

Critic：

```text
kinematic critic history 850
+ force context 12
+ privileged actual virtual force 6
= 868D
```

真实力不会进入 `act()` 或 `act_inference()` 的特征构造，只用于 value function
和 PPO update 中的 estimator Smooth-L1 label。

### Residual stochastic action

\[
z\sim\mathcal N(\mu,\sigma),\quad
a_{\mathrm{res}}=s\odot\tanh(z),\quad
a_{\mathrm{total}}=a_{\mathrm{base}}+a_{\mathrm{res}}
\]

- 腿和腰 residual scale 为 0；
- 14 个上肢 residual scale 为 0.5；
- inactive axes 不参加 log-prob、entropy、KL；
- RolloutStorage 保存 \(z\)，环境执行 \(a_\mathrm{total}\)；
- PPO replay 从原始 observation 重算 base action 和 estimated force，不保存
  stale estimate。

## 3. Virtual spring

每只手：

\[
F_\mathrm{ext}=K_p(x_s-x_g)+K_d(v_s-v_g)
\]

- `dt=0.005 s`；
- decimation=4；
- policy=50 Hz；
- spring=200 Hz；
- 默认 \(K_p=700\)、\(K_d=6\)，force clamp=20 N；
- spring force 只作用在 dataset z 轴；
- x/y 为 position axes。

`VirtualSpringJointPositionAction.apply_actions()` 的顺序是 joint target、
spring update、随后由原生环境执行 `scene.write_data_to_sim()` 和 PhysX step。
没有复制或改变基类 recorder/reset/command/event 生命周期。

施力 API 使用 world force、world application position、`is_global=True`。

## 4. Reference、坐标和符号

CSV sample 保存在 world frame，policy/reward 按需转换。policy step 时序：

1. Actor 读取 \(t\) 的 post-step observation；
2. physics 前 reference 切到 \(t+\Delta t\)；
3. spring 在四个 substep 读取相同移动 anchor，并用每个 substep 的最新手状态；
4. reward 在 \(t+\Delta t\) 的 post-physics state 计算；
5. observation 使用同一 reference。

MATLAB CSV：

\[
F_\mathrm{CSV}=F_\mathrm{hand\rightarrow payload}
\]

仿真/论文：

\[
F_\mathrm{cmd}=F_\mathrm{environment\rightarrow hand}=-F_\mathrm{CSV}
\]

当前每手 CSV `+0.24525 N` z 变成目标 `-0.24525 N` z。position/velocity
error 与 axis mask 在 reset-aligned dataset frame N 中计算；target/actual/
estimated force 给 policy 和 plot 时统一转换到 torso 或 world。

## 5. Reward 结构

S1 locomotion/stability reward 保留。S2 继续关闭：

- fixed arm target pose；
- total-action rate；
- imaginary torso-fixed box overlap；
- S0 arm-default deviation。

手部 reward：

| Reward | Weight | Scale |
|---|---:|---:|
| x/y position fine | 4.0 | 0.10 m |
| x/y position wide | 2.0 | 0.50 m |
| orientation fine | 1.0 | 0.30 rad |
| orientation wide | 1.0 | 1.00 rad |
| x/y linear velocity | 0.5 | 0.30 m/s |
| angular velocity | 0.25 | 0.50 rad/s |
| z virtual force fine | 5.0 | 0.25 N |
| z virtual force wide | 1.0 | 1.00 N |
| force clamp fraction | -0.10 | fraction |
| arm residual magnitude | -0.01 | L2 |
| arm residual rate | -0.02 | L2 |

force kernel 为论文的 vector norm exponential：

\[
\frac{1}{2}\sum_i\exp(-\|F_i-F^\mathrm{cmd}_i\|/\sigma)
\]

左右手分别算误差后平均，不能以两手合力相消。

## 6. Checkpoint、play 与 diagnostics

- policy schema：v5；
- raw deployable residual obs：219D；
- normalized control：233D；
- estimated force：6D；
- Actor：239D；
- Critic：868D；
- frozen S1 checkpoint 仍可用于新 S2 run；
- 所有旧 S2 checkpoint 不可 resume/play 新结构。

play tracking CSV/plot 区分：

- original CSV hand-on-payload force；
- target environment-on-hand force；
- actual applied virtual spring force；
- learned estimator force。

## 7. 尚存的实验边界

1. 当前 CSV 只提供恒定 z force；不能证明一般化 3D force command tracking。
2. actual force 是仿真 virtual-force truth，不是腕部传感器或接触力。
3. 尚未实现论文的多档随机 force command、ramp/hold/ramp-down、20% free
   spring-off 样本以及 Kp/Kd randomization。
4. 0.24525 N / 700 N/m 只有约 0.35 mm 平衡挠度；可能小于 G1 policy/PD 的
   有效学习尺度，必须通过运行时误差和 clamp fraction 验证。
5. 没有 moment tracking、真实 payload dynamics、grasp-loss termination。
6. 直线轨迹在 torso frame 近似常量，主要验证行走中保持/受力，不能证明复杂
   relative arm trajectory tracking。
7. residual normalizer 当前只支持单 GPU；distributed S2 被显式拒绝。
8. 未来旋转/曲线轨迹仍应加入 common object SE(3) alignment、lookahead 或
   phase，并验证 reference 在 physics-substep 间的插值。
