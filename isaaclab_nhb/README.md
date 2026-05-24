![](https://picgo-nanhaibei.oss-cn-beijing.aliyuncs.com/002%20dataset_management.png)

------

# Beyond Mimic Training Pipeline

本文档介绍了从原始 SMPL 数据到训练 Beyond Mimic 策略的完整流程。

## 1. 数据预处理 (Data Pre-processing)

数据处理主要依赖 **GMR (General Motion Retargeting)** 框架。请先部署 GMR 环境。

- **GMR Repository**: https://github.com/YanjieZe/GMR

### Step 1.1: SMPL 转 SMPL-X

将原始 SMPL 格式（72维 pose）转换为 GMR 兼容的 SMPL-X 格式。

- **脚本**: `scripts/smpl_to_smplx.py`

- **单文件处理**:

  ```Bash
  python smpl_to_smplx.py --input_file path/to/data.npz --output_file path/to/output.npz --gender neutral
  ```

- **批量处理**:

  ```Bash
  python smpl_to_smplx.py --src_folder path/to/smpl_dir --tgt_folder path/to/smplx_dir
  ```

### Step 1.2: Motion Retargeting (SMPL-X to Robot)

利用 GMR 将 SMPL-X 映射到机器人（如 Unitree G1）的关节空间。

- **单文件处理**: `scripts/smplx_to_robot.py`

  ```Bash
  python smplx_to_robot.py --smplx_file path/to/motion.npz --save_path path/to/robot_motion.pkl
  ```

- **批量处理**: `smplx_to_robot_dataset.py`

  - 请修改脚本中的 `src_folder` 和 `tgt_folder` 变量，或根据脚本参数运行：

  ```Bash
  python smplx_to_robot_dataset.py --src_folder path/to/smplx_dir --tgt_folder path/to/pkl_dir
  ```

  - **输出**: 包含机器人关节数据的 `.pkl` 文件。

### Step 1.3: 格式转换 (PKL to CSV)

将 GMR 输出的 `.pkl` 文件转换为 Beyond Mimic 数据处理所需的 `.csv` 格式。

- **脚本**: `scripts/batch_gmr_pkl_to_csv.py`

- **命令**:

  ```Bash
  python batch_gmr_pkl_to_csv.py --folder path/to/pkl_dir
  ```

- **输出**: 脚本会在输入目录下创建 `csv` 文件夹。

### Step 1.4: 物理仿真重放 (CSV to NPZ)

在 Isaac Sim 中重放 CSV 动作，利用物理引擎解算精确的速度、角速度和世界系状态。

- **脚本**: `isaaclab_nhb/script/utils/BeyondMimic/csv_to_npz.py`

- **命令**:

  ```Bash
  # 确保在 isaaclab_nhb/script/utils/BeyondMimic 目录下或相应路径
  python csv_to_npz.py --input_file path/to/motion.csv --output_name path/to/final_data.npz --input_fps 30 --output_fps 50
  ```

  *(注: 目前此脚本通常用于单文件处理，如需批量请自行编写 shell 循环)*

## 2. 训练 (Training)

数据准备好后（`.npz` 格式），即可开始训练。

### 修改训练数据路径

这是最关键的一步。你需要指定生成的 `.npz` 文件路径，否则训练将加载错误的动作或报错。

- **文件位置**: `isaaclab_nhb/tasks/humanoid/G1/G1_beyond_mimic_env_cfg.py`
- **修改位置**: `CommandsCfg` 类中的 `motion_file` 参数。

```python
@configclass
class CommandsCfg:
    # 动作文件路径将在运行时通过命令行参数覆盖，或者在此处硬编码调试
    motion = mdp_nhb.MotionCommandCfg(
        asset_name="robot",
        # ... 其他参数
        # 请修改这里指向你生成的 .npz 文件
        motion_file="/path/to/your/motion_data/final_data.npz", 
        # ...
    )
```

### 资产配置 (Asset Config)

- **文件位置**: `isaaclab_nhb/tasks/humanoid/G1/G1_beyond_mimic_asset_cfg.py`
- **核心内容**:
  1. **URDF 路径**: 指定了机器人模型的加载路径。
  2. **Actuator Gains (PD参数)**: 定义了不同关节组（腿、脚、腰、臂）的 `stiffness` (P) 和 `damping` (D)。
     - Beyond Mimic 甚至细化到了 `armature` (电枢惯量) 参数，这是为了更精确地模拟电机动力学。
  3. **Action Scale**: 计算了动作缩放比例 `G1_LEGACY_ACTION_SCALE`。
     - 逻辑是 `0.25 * effort / stiffness`。这意味着网络输出的 1.0 对应 1/4 的最大力矩所需的偏移量，这是一种常见的用于稳定训练的技巧。

------

### 任务详解 (Task Variants)

以 `BeyondMimic-G1-Flat` 为例：

```Python
gym.register(
    id="BeyondMimic-G1-Flat",  # 1. 任务 ID (命令行 --task 参数用这个)
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv", # 2. 环境入口类 (通常是自定义的 ManagerBasedRLEnv)
    disable_env_checker=True,
    kwargs={
        # 3. 环境配置类入口 (定义观测、奖励、场景等)
        "env_cfg_entry_point": f"{__name__}.G1_beyond_mimic_env_cfg:BeyondMimicG1FlatEnvCfg",
        # 4. PPO 算法配置类入口 (定义网络架构、超参数)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:BeyondMimicG1PPORunnerCfg",
    },
)
```

在 `__init__.py` 和配置文件中，定义了几个不同的 Beyond Mimic 任务变体，它们的区别如下：

#### A. `BeyondMimic-G1-Flat` (标准版)

- **对应配置**: `BeyondMimicG1FlatEnvCfg`
- **特点**: 包含完整的观测空间。
- **用途**: 标准训练基准。

#### B. `BeyondMimic-G1-Flat-Wo-State-Estimation` (无状态估计版)

- **对应配置**: `BeyondMimicG1FlatWoStateEstimationEnvCfg`

- **关键代码差异**:

  ```python
  class BeyondMimicG1FlatWoStateEstimationEnvCfg(BeyondMimicG1FlatEnvCfg):
      def __post_init__(self):
          super().__post_init__()
          # 移除“动作锚点位置”观测
          self.observations.policy.motion_anchor_pos_b = None
          # 移除“基座线速度”观测
          self.observations.policy.base_lin_vel = None
  ```

- 用途：无需线速度，只需要 IMU 即可部署

#### C. `BeyondMimic-G1-Flat-Low-Freq` (低频控制版)

- **对应配置**: `BeyondMimicG1FlatLowFreqEnvCfg`

- **关键代码差异**:

  ```Python
  self.decimation = round(self.decimation / LOW_FREQ_SCALE) # 增大降采样倍数 -> 降低控制频率
  ```

- **含义**: 用于测试策略在低控制频率（例如 25Hz 或 50Hz，而不是标准的 100Hz+）下的表现，或者模拟算力受限的机载电脑环境。

------

### 训练

```Bash
# 切换到 rsl_rl 脚本目录
cd isaaclab_nhb/script/rsl_rl

# 启动训练
python train.py --task BeyondMimic-G1-Flat --headless --num_envs 4096
```

#### 常用参数

- `--task`: 任务名称（如 `BeyondMimic-G1-Flat`, `BeyondMimic-G1-Flat-Wo-State-Estimation`）。
- `--headless`: 无图形界面模式（服务器训练必备）。
- `--video`: 开启视频录制。

------



# AMP Training Pipeline

本文档介绍了从原始 SMPL 数据到训练 AMP (Adversarial Motion Priors) 策略的完整流程。

## 1. 数据预处理 (Data Pre-processing)

### Step 1.1 & Step 1.2

前两步与 Beyond Mimic 流程相同（请参考上方文档）：

1. **SMPL to SMPL-X**: 使用 `smpl_to_smplx.py`。
2. **Retargeting**: 使用 `smplx_to_robot.py` 得到 `.pkl` 文件。

### Step 1.3: 格式转换 (PKL to Visualization TXT)

将 `.pkl` 文件转换为 AMP 可视化所需的中间 JSON 格式（`.txt`）。

- **脚本**: `isaaclab_nhb/script/utils/AMP/gmr_data_conversion.py`

  - 命令行传参

    - **单文件处理**:

      ```Bash
      # 确保在 isaaclab_nhb/script/utils/AMP 目录下或相应路径
      python gmr_data_conversion.py --input_pkl path/to/motion.pkl --output_txt path/to/vis_motion.txt 
      ```

    - **批量处理**:

    ```Bash
    # 确保在 isaaclab_nhb/script/utils/AMP 目录下或相应路径
    # 如果输入是文件夹，脚本会自动遍历并保持目录结构
    python gmr_data_conversion.py --input_pkl path/to/pkl_dir --output_dir path/to/txt_dir
    ```

  - **修改配置，直接运行**:

    1. 打开 `isaaclab_nhb/dataset/amp_data_cfg/G1_amp_data_cfg.py`。
    2. 修改 `gmr_data_conversion_input` 指向 Step 1.2 生成的 `.pkl` 文件夹。
    3. 修改 `visualization_path` 指定输出 `.txt` 的文件夹。

    ```python
    # 确保在 isaaclab_nhb/script/utils/AMP 目录下或相应路径
    python gmr_data_conversion.py
    ```

### Step 1.4: 生成专家数据 (Visualization to Expert Data)

在 Isaac Sim 中播放可视化数据，通过物理环境解算末端执行器（手/脚）的相对位置，并重新排列关节顺序，生成最终供 AMP 训练使用的专家数据。

- **脚本**: `play_amp_animation.py`
- **配置文件**: `isaaclab_nhb/dataset/amp_data_cfg/G1_amp_data_cfg.py`

#### 修改配置

1. **路径设置**:
   - `visualization_path`: 指向 Step 1.3 生成的 `.txt` **文件夹** (脚本会自动读取该目录下所有 txt)。
   - `save_motion_expert_path`: 指定最终专家数据的**输出文件路径** (生成模式下生效)。
2. **功能开关**:
   - **`save_motion_expert_amp_data`**: **导出控制 (关键!)**
     - `False` (默认): **预览模式**。仅在 Isaac Sim 中播放动画，方便肉眼检查动作质量，**不会保存文件**。
     - `True`: **生成模式**。播放结束后，会自动将包含 EndEffector 信息的完整数据写入磁盘。**正式训练前必须设为 True 运行一次**。
   - **`print_joint_order_with_end_order`**: **调试关节顺序**
     - `True`: 在控制台打印当前加载机器人的 Joint Order 和 End Effector Order。用于检查与 AMP 训练配置是否对齐。
   - **`print_frame_root_info`**: **调试根节点**
     - `True`: 在控制台逐帧打印 Root 的位置和旋转信息。用于检查数据是否存在瞬移或 NaN 异常。

#### 运行命令

```Bash
# 确保在 isaaclab_nhb/script/utils/AMP 目录下或相应路径
python play_amp_animation.py
```

- **输出**: 包含 `EndEffectorOrder`、`JointOrder` 以及完整帧数据（Pos + Vel + EndEffectorPos）的专家数据文件（`.txt`）。

## 2. 训练 (Training)

确保 `G1_amp_data_cfg.py` 中的 `motion_file` 或相关配置指向了 Step 1.4 生成的专家数据。

### 训练命令

```Bash
# 切换到 rsl_rl 脚本目录
cd isaaclab_nhb/script/rsl_rl

# 启动 AMP 训练
python train.py --task G1-AMP-Walk-Flat --headless --num_envs 4096
```

### 注意事项

- **环境数量**: AMP 对样本多样性要求较高，建议 `num_envs` 设置为 2048 或 4096。