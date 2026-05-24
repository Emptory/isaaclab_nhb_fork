import csv
import signal
from datetime import datetime
import os
import torch
from tensordict import TensorDict


class DataLogger:
    def __init__(self, log_dir, env, log_height_scan=True, exclude_obs_groups=None):
        """初始化数据记录器
        
        Args:
            log_dir: 日志保存目录
            env: 环境实例 (RslRlVecEnvWrapper 或 RslRlVecEnvWrapperDictAction)
            log_height_scan: 是否记录高度扫描数据，默认为True（已废弃，使用exclude_obs_groups代替）
            exclude_obs_groups: 要排除的观测组名称列表，例如['height_scan_policy', 'height_scan_critic', 'amp']
        """
        self.env = env
        self.log_height_scan = log_height_scan
        self.exclude_obs_groups = exclude_obs_groups if exclude_obs_groups is not None else []

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"robot_data_{timestamp}.csv")
        
        # 创建CSV文件并写入表头
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 获取observation和action的维度用于验证
        obs_dict = self.env.get_observations()
        action_dim = self.env.num_actions
        
        # 硬编码表头 - 基于G1ElevationNetMode13A5ObsCfg配置
        header = self._get_hardcoded_header()
        
        self.csv_writer.writerow(header)
        self.timestep = 0
        
        # 计算总观察维度（排除exclude_obs_groups中的项）
        obs_dim = sum(value.shape[-1] for key, value in obs_dict.items() 
                    if value.dim() > 0 
                    and key not in self.exclude_obs_groups)
        
        print(f"[INFO] Data logger initialized. Saving to: {self.csv_path}")
        print(f"[INFO] Observation dim: {obs_dim}, Action dim: {action_dim}")
        print(f"[INFO] Observation keys: {list(obs_dict.keys())}")
        if self.exclude_obs_groups:
            print(f"[INFO] Excluded observation groups: {self.exclude_obs_groups}")
        print(f"[INFO] Using hardcoded header based on G1ElevationNetMode13A5ObsCfg")
        
    def _get_hardcoded_header(self):
        """返回硬编码的CSV表头 - 基于G1ElevationNetMode13A5ObsCfg配置"""
        
        # G1机器人29个关节的顺序
        G1_29DOF_JOINT_ORDER = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint", 
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint", 
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        
        header = ['timestep']
        
        # Policy组观测 (96维)
        # 1. pelvis角速度 [3]
        header.extend([
            'base_ang_vel_x',
            'base_ang_vel_y', 
            'base_ang_vel_z'
        ])
        
        # 2. 重力投影 [3]
        header.extend([
            'projected_gravity_x',
            'projected_gravity_y',
            'projected_gravity_z'
        ])
        
        # 3. 关节位置 [29]
        header.extend([f'joint_pos_{name}' for name in G1_29DOF_JOINT_ORDER])
        
        # 4. 关节速度 [29]
        header.extend([f'joint_vel_{name}' for name in G1_29DOF_JOINT_ORDER])
        
        # 5. 速度命令 [3]
        header.extend([
            'velocity_commands_x',
            'velocity_commands_y',
            'velocity_commands_yaw'
        ])
        
        # 6. 上一帧动作 [29]
        header.extend([f'actions_{name}' for name in G1_29DOF_JOINT_ORDER])
        
        # Critic组观测 (99维)
        # 1. pelvis角速度 [3]
        header.extend([
            'critic_base_ang_vel_x',
            'critic_base_ang_vel_y', 
            'critic_base_ang_vel_z'
        ])
        
        # 2. 重力投影 [3]
        header.extend([
            'critic_projected_gravity_x',
            'critic_projected_gravity_y',
            'critic_projected_gravity_z'
        ])
        
        # 3. 关节位置 [29]
        header.extend([f'critic_joint_pos_{name}' for name in G1_29DOF_JOINT_ORDER])
        
        # 4. 关节速度 [29]
        header.extend([f'critic_joint_vel_{name}' for name in G1_29DOF_JOINT_ORDER])
        
        # 5. pelvis线速度 [3] - Critic独有
        header.extend([
            'critic_base_lin_vel_x',
            'critic_base_lin_vel_y',
            'critic_base_lin_vel_z'
        ])
        
        # 6. 速度命令 [3]
        header.extend([
            'critic_velocity_commands_x',
            'critic_velocity_commands_y',
            'critic_velocity_commands_yaw'
        ])
        
        # 7. 上一帧动作 [29]
        header.extend([f'critic_actions_{name}' for name in G1_29DOF_JOINT_ORDER])
        
        # 动作 [29]
        header.extend([f'action_{name}' for name in G1_29DOF_JOINT_ORDER])
        
        return header
    
    def log(self, obs, actions):
        """记录一帧数据
        
        Args:
            obs: 观察值 (TensorDict)
            actions: 动作张量
        """
        # 只记录第0个环境的数据
        # 处理TensorDict格式的观察
        obs_data = []
        for key in obs.keys():
            # 跳过在排除列表中的观测组
            if key in self.exclude_obs_groups:
                continue
                
            obs_tensor = obs[key]
            if obs_tensor.dim() > 0:
                obs_0 = obs_tensor[0].cpu().numpy()
                if obs_tensor.shape[-1] == 1:
                    obs_data.append(obs_0.item())
                else:
                    obs_data.extend(obs_0.tolist())
        
        action_0 = actions[0].cpu().numpy()
        
        # 组合数据
        row = [self.timestep] + obs_data + action_0.tolist()
        self.csv_writer.writerow(row)
        
        self.timestep += 1
        
    def close(self):
        """关闭文件"""
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
            print(f"\n[INFO] Data logged to: {self.csv_path}")
            print(f"[INFO] Total timesteps recorded: {self.timestep}")
