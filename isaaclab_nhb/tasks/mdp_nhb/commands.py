# 自己写的命令值
from __future__ import annotations
import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from isaaclab_nhb.utils import pickle
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .commands_cfg import (
        BipedalGaitCommandCfg, 
        QuadrupedGaitCommandCfg, 
        MimicCommandCfg, 
        TerrainAdaptiveVelocityCommandCfg
    )


class BipedalGaitCommand(CommandTerm):

    cfg: BipedalGaitCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: BipedalGaitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 命令值 [stance_rate, bipedal_offset, gait_frequency, sin_l, cos_l, sin_r, cos_r]
        self.gait_command = torch.zeros(self.num_envs, 7, device=self.device) 

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "SCCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    """
    Properties
    """
    
    @property
    def command(self) -> torch.Tensor:
        """Return the command tensor."""
        return self.gait_command
    
    """
    Implementation specific functions.
    """
    
    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """
        命令重采样
        训练刚开始环境初始化时会对所有环境进行一次重采样
        """

        r = torch.empty(len(env_ids), device=self.device)

        # 获取速度命令值
        vel_cmd = self._env.command_manager.get_command("base_velocity")
        right_vel_flag = vel_cmd[env_ids,1] > 0.0 # 如果有右平移的速度
        stance_flag = torch.norm(vel_cmd[env_ids, :2],dim=1) < 0.1 

        # stance_rate
        # self.gait_command[env_ids,0] = r.uniform_(*self.cfg.ranges.stance_rate)
        self.gait_command[env_ids,0] = torch.where(
            stance_flag,
            torch.ones_like(self.gait_command[env_ids,0]),
            r.uniform_(*self.cfg.ranges.stance_rate)
        )
        # bipedal_offset
        self.gait_command[env_ids,1] = r.uniform_(*self.cfg.ranges.bipedal_offset)
        self.gait_command[env_ids,1] = torch.where( # 有右平移速度时，右脚在前
            right_vel_flag,
            self.gait_command[env_ids,1] * -1.0,
            self.gait_command[env_ids,1]
        )
        # gait_frequency
        # self.gait_command[env_ids,2] = r.uniform_(*self.cfg.ranges.gait_frequency)
        self.gait_command[env_ids,2] = torch.where(
            stance_flag,
            torch.zeros_like(self.gait_command[env_ids,2]),
            r.uniform_(*self.cfg.ranges.gait_frequency)
        )


    
    def _update_command(self):
        # 获得当前相位
        self.phase = (self._env.episode_length_buf * self._env.step_dt * self.gait_command[:, 2]) % 1.0

        # 更新正余弦命令
        self.gait_command[:, 3] = torch.sin(2 * torch.pi * self.phase)
        self.gait_command[:, 4] = torch.cos(2 * torch.pi * self.phase)
        self.gait_command[:, 5] = torch.sin(2 * torch.pi * (self.phase + self.gait_command[:, 1])) # 偏差以百分比表示，加在这是对的
        self.gait_command[:, 6] = torch.cos(2 * torch.pi * (self.phase + self.gait_command[:, 1]))

class QuadrupedGaitCommand(CommandTerm):

    cfg: QuadrupedGaitCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: QuadrupedGaitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 命令值 [stance_rate, rf_offset, lb_offset, rb_offset, gait_frequency, sin_lf, cos_lf, sin_rf, cos_rf, sin_lb, cos_lb, sin_rb, cos_rb]
        self.gait_command = torch.zeros(self.num_envs, 13, device=self.device) 

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "SCCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    """
    Properties
    """
    
    @property
    def command(self) -> torch.Tensor:
        """Return the command tensor."""
        return self.gait_command
    
    """
    Implementation specific functions.
    """
    
    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """
        命令重采样
        环境初始化时会对所有环境进行一次重采样
        """

        r = torch.empty(len(env_ids), device=self.device)

        # 获取速度命令值
        vel_cmd = self._env.command_manager.get_command("base_velocity")
        right_vel_flag = vel_cmd[env_ids,1] > 0.0 # 如果有右平移的速度
        back_vel_flag = vel_cmd[env_ids,0] < 0.0
        stance_flag = torch.max(torch.abs(vel_cmd[env_ids, :]),dim=1).values < 0.1 # 获取三个方向中的最大值进行比较

        # stance_rate
        # self.gait_command[env_ids,0] = r.uniform_(*self.cfg.ranges.stance_rate)
        self.gait_command[env_ids,0] = torch.where(
            stance_flag,
            torch.ones_like(self.gait_command[env_ids,0]),
            r.uniform_(*self.cfg.ranges.stance_rate)
        )
        # rf_offset
        self.gait_command[env_ids,1] = r.uniform_(*self.cfg.ranges.rf_offset)
        self.gait_command[env_ids,1] = torch.where(
            right_vel_flag | back_vel_flag,
            self.gait_command[env_ids,1] * -1.0,
            self.gait_command[env_ids,1]
        )
        # lb_offset
        self.gait_command[env_ids,2] = r.uniform_(*self.cfg.ranges.lb_offset)
        self.gait_command[env_ids,2] = torch.where(
            right_vel_flag | back_vel_flag,
            self.gait_command[env_ids,2] * -1.0,
            self.gait_command[env_ids,2]
        )
        # rb_offset
        self.gait_command[env_ids,3] = r.uniform_(*self.cfg.ranges.rb_offset)
        # gait_cycle
        # self.gait_command[env_ids,4] = r.uniform_(*self.cfg.ranges.gait_frequency)
        self.gait_command[env_ids,4] = torch.where(
            stance_flag,
            torch.zeros_like(self.gait_command[env_ids,4]),
            r.uniform_(*self.cfg.ranges.gait_frequency)
        )
    
    def _update_command(self):
        # 获得当前相位
        self.phase = (self._env.episode_length_buf * self._env.step_dt * self.gait_command[:, 4]) % 1.0

        # 正余弦命令-lf
        self.gait_command[:, 5] = torch.sin(2 * torch.pi * self.phase)
        self.gait_command[:, 6] = torch.cos(2 * torch.pi * self.phase)
        # 正余弦命令-rf
        self.gait_command[:, 7] = torch.sin(2 * torch.pi * (self.phase + self.gait_command[:, 1])) # 偏差以百分比表示，加在这是对的
        self.gait_command[:, 8] = torch.cos(2 * torch.pi * (self.phase + self.gait_command[:, 1]))
        # 正余弦命令-lb
        self.gait_command[:, 9] = torch.sin(2 * torch.pi * (self.phase + self.gait_command[:, 2])) 
        self.gait_command[:, 10] = torch.cos(2 * torch.pi * (self.phase + self.gait_command[:, 2]))
        # 正余弦命令-rb
        self.gait_command[:, 11] = torch.sin(2 * torch.pi * (self.phase + self.gait_command[:, 3])) 
        self.gait_command[:, 12] = torch.cos(2 * torch.pi * (self.phase + self.gait_command[:, 3]))

class MimicCommand(CommandTerm):
    """模仿命令。"""

    def __init__(self, cfg: MimicCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # 加载主要数据集
        data = pickle.p_load(cfg.data_path)
        self.dataset = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in data.items()}
        # 改变数据中某些关节的正负号
        self.dataset["joint_vel"][:,2] = self.dataset["joint_vel"][:,2] * -1.0 
        self.dataset["joint_vel"][:,3] = self.dataset["joint_vel"][:,3] * -1.0 
        self.dataset["joint_vel"][:,8] = self.dataset["joint_vel"][:,8] * -1.0 
        self.dataset["joint_vel"][:,9] = self.dataset["joint_vel"][:,9] * -1.0 

        self.dataset["joint_pos"][:,2] = self.dataset["joint_pos"][:,2] * -1.0 
        self.dataset["joint_pos"][:,3] = self.dataset["joint_pos"][:,3] * -1.0 
        self.dataset["joint_pos"][:,8] = self.dataset["joint_pos"][:,8] * -1.0 
        self.dataset["joint_pos"][:,9] = self.dataset["joint_pos"][:,9] * -1.0 
        idx = torch.cat([torch.arange(0, 12, 2), torch.arange(1, 12, 2)])
        self.dataset["joint_pos"] = self.dataset["joint_pos"][:, idx]
        self.dataset["joint_vel"] = self.dataset["joint_vel"][:, idx]
        
        # 加载Y轴数据
        def load_data_flexible(file_path):
            """
            加载数据文件，支持 pickle 和 npz 格式
            Args:
                file_path: 数据文件的完整路径（包含扩展名）
            Returns:
                data: 加载的数据字典
            """
            # 首先检查文件格式是否支持
            if not (file_path.endswith('.npz') or file_path.endswith('.pickle')):
                raise ValueError(f"Unsupported file format: {file_path}. Supported formats: .npz, .pickle")
            
            # 根据文件扩展名决定加载方式
            if file_path.endswith('.npz'):
                data_npz = np.load(file_path)
                # 将 npz 数据转换为字典格式
                data = {}
                for key in data_npz.files:
                    data[key] = data_npz[key]
                return data
            elif file_path.endswith('.pickle'):
                # 去掉 .pickle 扩展名，因为 pickle.p_load 会自动添加
                base_path = file_path[:-7]  # 去掉 '.pickle'
                return pickle.p_load(base_path)
        
        y_axis_path = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/dataset/S3-lbl/Y_axis_locomotion_2.4s_50Hz.npz"
        y_axis_data = load_data_flexible(y_axis_path)
        self.y_axis_dataset = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in y_axis_data.items()}
        
        self.len_dataset = self.dataset["joint_pos"].shape[0]
        self.len_y_axis_dataset = self.y_axis_dataset["joint_pos_1"].shape[0]
        self.idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.y_axis_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.end = self.idx == (self.len_dataset - 1)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitsCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        # 获取主要数据集的关节位置
        joint_pos = self.dataset["joint_pos"][self.idx].clone()
        # 设置Y轴数据：0号数据等于joint_pos_1，5号数据等于joint_pos_2
        joint_pos[:, 0] = self.y_axis_dataset["joint_pos_1"][self.y_axis_idx]
        joint_pos[:, 5] = self.y_axis_dataset["joint_pos_2"][self.y_axis_idx]
        return joint_pos

    def get_data(self, name) -> torch.Tensor:
        if name in self.y_axis_dataset:
            return self.y_axis_dataset[name][self.y_axis_idx]
        return self.dataset[name][self.idx]

    def _resample_command(self, env_ids: Sequence[int]):
        self.idx[env_ids] = (
            torch.randint(0, self.len_dataset - 1, (len(env_ids),), dtype=torch.long, device=self.device) * 0
        )
        self.y_axis_idx[env_ids] = 0

    def _update_command(self):
        self.idx += 1
        self.y_axis_idx += 1
        
        # 循环主要数据集
        self.idx %= self.len_dataset
        self.end = self.idx == (self.len_dataset - 1)
        
        # 循环Y轴数据集
        self.y_axis_idx %= self.len_y_axis_dataset

    def _update_metrics(self):
        pass

class QuadrupedMimicCommand(CommandTerm):
    """四足机器人模仿命令。"""

    def __init__(self, cfg: MimicCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # 加载数据的辅助函数
        def load_data_flexible(file_path):
            """
            加载数据文件，支持 pickle 和 npz 格式
            Args:
                file_path: 数据文件的完整路径（包含扩展名）
            Returns:
                data: 加载的数据字典
            """
            if not (file_path.endswith('.npz') or file_path.endswith('.pickle')):
                raise ValueError(f"Unsupported file format: {file_path}. Supported formats: .npz, .pickle")
            
            if file_path.endswith('.npz'):
                data_npz = np.load(file_path)
                data = {}
                for key in data_npz.files:
                    data[key] = data_npz[key]
                return data
            elif file_path.endswith('.pickle'):
                base_path = file_path[:-7]  # 去掉 '.pickle'
                return pickle.p_load(base_path)
        
        # 加载Go2四足机器人数据集
        go2_data = load_data_flexible(cfg.data_path)
        self.dataset = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in go2_data.items()}
        
        # Go2关节顺序: FL(3) | FR(3) | RL(3) | RR(3)
        # 每组: hip | thigh | calf
        # 数据已经是正确的顺序，不需要像双足那样复杂的调整
        
        self.len_dataset = self.dataset["joint_pos"].shape[0]
        self.idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.end = self.idx == (self.len_dataset - 1)
        
        print(f"[QuadrupedMimicCommand] Loaded dataset with {self.len_dataset} frames")
        print(f"[QuadrupedMimicCommand] Joint positions shape: {self.dataset['joint_pos'].shape}")

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "QuadrupedMimicCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tDataset length: {self.len_dataset} frames"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """返回当前帧的关节位置命令"""
        return self.dataset["joint_pos"][self.idx]

    def get_data(self, name: str) -> torch.Tensor:
        """获取指定名称的数据"""
        if name not in self.dataset:
            raise KeyError(f"Data '{name}' not found in dataset. Available keys: {list(self.dataset.keys())}")
        return self.dataset[name][self.idx]

    def _resample_command(self, env_ids: Sequence[int]):
        """
        命令重采样
        可以选择从随机位置开始或从头开始
        """
        # 从头开始播放
        self.idx[env_ids] = 0
        
        # 如果想要随机起始位置，可以使用：
        # self.idx[env_ids] = torch.randint(0, self.len_dataset, (len(env_ids),), dtype=torch.long, device=self.device)

    def _update_command(self):
        """
        更新命令，每步前进一帧
        循环播放数据集
        """
        self.idx += 1
        
        # 循环播放
        self.idx %= self.len_dataset
        self.end = self.idx == (self.len_dataset - 1)

    def _update_metrics(self):
        pass

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found at: {motion_file}")

        print(f"[DEBUG] Loading motion from: {motion_file}")
        data = np.load(motion_file)

        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        # --- [DEBUG] 打印 _body_pos_w 的原始形状 ---
        raw_pos = data["body_pos_w"]
        print(f"\n{'=' * 40}")
        print(f"[DEBUG] MotionLoader Data Analysis")
        print(f"{'=' * 40}")
        print(f"File '_body_pos_w' shape: {raw_pos.shape}")
        print(f"  -> Time steps: {raw_pos.shape[0]}")
        print(f"  -> Num Bodies in File: {raw_pos.shape[1]}")  # 这里是文件包含的刚体数量
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        # --- [DEBUG] 检查索引是否越界 ---
        self._body_indexes = body_indexes
        print(f"Requested Indices: {self._body_indexes}")

        # 将 Sequence 转为 list 或 tensor 方便计算最大值
        if isinstance(body_indexes, torch.Tensor):
            max_req_idx = body_indexes.max().item()
        else:
            max_req_idx = max(body_indexes)

        file_body_count = self._body_pos_w.shape[1]

        if max_req_idx >= file_body_count:
            print(f"\n[CRITICAL WARNING] Index Out of Bounds Detected!")
            print(f"  -> You are requesting index: {max_req_idx}")
            print(f"  -> But the file only has:    {file_body_count} bodies")
            print(f"  -> 访问 self.body_pos_w 时程序将会崩溃 (CUDA error)")
        else:
            print(f"[OK] Max requested index {max_req_idx} is within file limit {file_body_count}")

        print(f"{'=' * 40}\n")
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # ================== 强力调试打印区 START ==================
        print(f"\n{'=' * 50}")
        print(f"[DEBUG] Robot Body Count Analysis")
        print(f"{'=' * 50}")

        # 1. 打印物理引擎 Buffer 中的刚体（这是真相，索引绝对不能超过这个数量）
        # 强制刷新确保数据加载
        if self.robot.data.body_pos_w is None or self.robot.data.body_pos_w.shape[0] == 0:
            self.robot.update(0.0)

        sim_body_names = self.robot.data.body_names
        sim_count = len(sim_body_names)
        print(f"[TRUTH] Physics Tensor Shape: {self.robot.data.body_pos_w.shape}")
        print(f"[TRUTH] Physics Body Count:   {sim_count}")
        print(f"[TRUTH] Physics Body Names:   \n{sim_body_names}")
        print(f"{'-' * 50}")

        # 2. 打印你请求的刚体索引
        found_bodies_indices = self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0]
        print(f"[REQUEST] Config Body Names: {self.cfg.body_names}")
        print(f"[REQUEST] Found Indices:     {found_bodies_indices}")

        # 3. 现场抓获越界索引
        print(f"{'-' * 50}")
        print(f"[ANALYSIS] Checking for Out-of-Bounds Indices...")
        safe_indices = []
        has_error = False
        for i, idx in enumerate(found_bodies_indices):
            name = self.cfg.body_names[i]
            if idx >= sim_count:
                print(
                    f"  [!!! CRITICAL ERROR !!!] Body '{name}' has index {idx}, but Physics only has {sim_count} bodies!")
                print(f"  -> This IS the cause of the crash. Clamping to 0.")
                safe_indices.append(0)
                has_error = True
            else:
                print(f"  [OK] Body '{name}' index {idx} is valid.")
                safe_indices.append(idx)

        if not has_error:
            print("[ANALYSIS] All indices look valid. If it still crashes, it's a miracle.")
        else:
            print("[ANALYSIS] Indices clamped. Crash should be prevented.")

        print(f"{'=' * 50}\n")
        # ================== 强力调试打印区 END ==================


        # # 打印配置中请求追踪的刚体名称
        # print(f"\n[DEBUG] >>> MotionCommand Init")
        # print(f"[DEBUG] Configured anchor_body_name: {self.cfg.anchor_body_name}")
        # print(f"[DEBUG] Configured body_names to track: {self.cfg.body_names}")

        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)

        # 获取这些刚体在仿真机器人中的索引
        found_bodies_indices = self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0]
        print(f"[DEBUG] Found robot body indices: {found_bodies_indices}")

        # --- [关键修复] 安全检查：确保索引不越界 ---
        max_robot_idx = max(found_bodies_indices)
        if max_robot_idx >= self.robot.num_bodies:
            print(f"\n[ERROR] CRITICAL INDEX MISMATCH DETECTED!")
            print(
                f"Requesting index {max_robot_idx}, but robot only has {self.robot.num_bodies} bodies in data buffer.")
            print(f"This causes the CUDA 'index out of bounds' crash.")
            # 强制修正索引防止崩溃（虽然数据可能是错的，但至少能跑起来看到 log）
            # 这里我们把它 clamp 到最大有效值
            found_bodies_indices = [min(idx, self.robot.num_bodies - 1) for idx in found_bodies_indices]
            print(f"[WARNING] Indices clamped to: {found_bodies_indices}\n")

        self.body_indexes = torch.tensor(
            found_bodies_indices, dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        self.time_steps[env_ids] = (sampled_bins / self.bin_count * (self.motion.time_step_total - 1)).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

class TerrainAdaptiveVelocityCommand(UniformVelocityCommand):
    """基于地形类型和难度的自适应速度命令
    
    继承自UniformVelocityCommand，保留所有原有功能（heading控制、standing环境、metrics、debug可视化等）。
    额外增加：当机器人处于concentric_moats地形时，根据难度等级动态调整速度命令范围。
    其他地形使用默认速度范围。
    """
    
    cfg: TerrainAdaptiveVelocityCommandCfg
    
    def __init__(self, cfg: TerrainAdaptiveVelocityCommandCfg, env: ManagerBasedEnv):
        # 调用父类初始化（会设置所有必要的buffers和metrics）
        super().__init__(cfg, env)
        
        # 获取terrain信息
        self.terrain = self._env.scene.terrain
        
        # 确定concentric_moats地形在terrain_types中的索引
        if hasattr(self.terrain, 'terrain_origins') and self.terrain.terrain_origins is not None:
            # 获取地形生成器配置
            terrain_gen_cfg = self._env.cfg.scene.terrain.terrain_generator
            if terrain_gen_cfg is not None:
                sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
                self.concentric_moats_idx = next((i for i, name in enumerate(sub_terrain_names) if 'concentric_moats' in name.lower()), None)
            else:
                self.concentric_moats_idx = None
        else:
            self.concentric_moats_idx = None
    
    def __str__(self) -> str:
        msg = "TerrainAdaptiveVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tConcentric moats terrain index: {self.concentric_moats_idx}\n"
        msg += f"\tMoats speed threshold: {self.cfg.moats_min_speed_threshold}"
        return msg
    
    def _resample_command(self, env_ids: Sequence[int]):
        """根据地形类型和难度重采样速度命令（向量化版本）
        
        对于concentric_moats地形：
        - 难度低时: 在整个范围内采样（包括低速）
        - 难度高时: 只在高速区域采样，排除中间低速区域
        例如: 范围[-1.2, 1.2]，难度9时只采样[-1.2, -1.0]和[1.0, 1.2]
        
        其他地形使用标准的均匀采样（与UniformVelocityCommand相同）。
        """
        if len(env_ids) == 0:
            return
        
        num_resampled = len(env_ids)
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # 获取每个环境的地形类型和难度等级
        if hasattr(self.terrain, 'terrain_types') and hasattr(self.terrain, 'terrain_levels'):
            terrain_types = self.terrain.terrain_types[env_ids_tensor]
            terrain_levels = self.terrain.terrain_levels[env_ids_tensor]
            max_level = self.terrain.max_terrain_level
        else:
            terrain_types = None
            terrain_levels = None
            max_level = 1
        
        # 批量判断是否是concentric_moats地形
        is_moats_terrain = torch.zeros(num_resampled, dtype=torch.bool, device=self.device)
        if terrain_types is not None and self.concentric_moats_idx is not None:
            is_moats_terrain = (terrain_types == self.concentric_moats_idx)
        
        # 计算难度比例和排除比例
        difficulty_ratio = torch.zeros(num_resampled, device=self.device)
        if terrain_levels is not None:
            difficulty_ratio = terrain_levels.float() / max(max_level - 1, 1)
        excluded_ratio = difficulty_ratio * self.cfg.moats_min_speed_threshold
        
        # 准备采样
        r = torch.empty(num_resampled, device=self.device)
        
        # 准备速度范围
        vel_ranges = torch.tensor([
            [self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1]],
            [self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1]],
            [self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1]]
        ], device=self.device)  # [3, 2]
        
        # 批量采样速度（对于moats地形使用排除中间区域的采样，其他地形使用均匀采样）
        for vel_idx in range(3):
            vel_min, vel_max = vel_ranges[vel_idx]
            abs_max = max(abs(vel_min), abs(vel_max))
            
            # 计算每个环境的阈值
            threshold = abs_max * excluded_ratio  # [num_resampled]
            
            # 随机选择正向或负向
            direction_mask = torch.rand(num_resampled, device=self.device) < 0.5
            
            # 计算采样范围的下界和上界
            # 对于moats地形：负向[vel_min, -threshold]，正向[threshold, vel_max]
            # 对于其他地形：统一[vel_min, vel_max]
            lower_bound = torch.where(
                is_moats_terrain & direction_mask,
                torch.full((num_resampled,), vel_min, device=self.device),
                torch.where(
                    is_moats_terrain,
                    threshold,
                    torch.full((num_resampled,), vel_min, device=self.device)
                )
            )
            
            upper_bound = torch.where(
                is_moats_terrain & direction_mask,
                -threshold,
                torch.full((num_resampled,), vel_max, device=self.device)
            )
            
            # 批量采样
            sampled_vel = lower_bound + torch.rand(num_resampled, device=self.device) * (upper_bound - lower_bound)
            self.vel_command_b[env_ids_tensor, vel_idx] = sampled_vel
        
        # 采样heading目标（如果启用了heading命令）
        if self.cfg.heading_command:
            self.heading_target[env_ids_tensor] = r.uniform_(*self.cfg.ranges.heading)
            # 更新heading环境标记
            self.is_heading_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        # 更新standing环境标记
        self.is_standing_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
