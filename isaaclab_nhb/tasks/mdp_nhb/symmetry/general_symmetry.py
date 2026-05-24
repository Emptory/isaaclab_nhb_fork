"""写一个函数兼容多种机器人的对称性数据增强"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING
from tensordict import TensorDict
from pathlib import Path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    # from omni.isaac.lab.envs import ManagerBasedRLEnv

# 限制 [import *]的导入范围
__all__ = ["compute_symmetric_states"]

# 合理的观测值项
valid_obs_term = [
    "base_ang_vel",
    "torso_ang_vel",
    "projected_gravity",
    "torso_projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "gait_commands",
    "gait_commands_quadruped",
    "height_scan",
]

# 合理的机器人类型
valid_robot_name = [
    "G1_12dof",
    "G1_29dof",
    "S3_22dof",
    "L1",
    "Galileo",
    "Go2"
]
# TODO:尚未适配四足的前后、斜角对称

class Compute_symmetric_states():
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        # 检查机器人名称是否合理并获取机器人名称
        robot_usd_path = env.cfg.scene.robot.spawn.usd_path
        usd_name = Path(robot_usd_path).name # 获取usd文件名
        matches = [name for name in valid_robot_name if name in usd_name] # 单行查找匹配
        if not matches:
            raise ValueError(f"No robot found in {usd_name}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple robots found: {matches}")
        else:
            self.robot_name = matches[0]

        # 获取观测值项的名称和维度
        obs_name_list = env.env.env.observation_manager.active_terms['policy']
        obs_dim_list = env.env.env.observation_manager.group_obs_term_dim['policy']
        self.policy_obs_terms = []
        order_index = 0
        
        # 遍历观测项名称列表和对应的维度列表
        for obs_name, obs_dim in zip(obs_name_list, obs_dim_list):
            # 检查观测值项是否合法
            if obs_name not in valid_obs_term:
                raise ValueError(f"obs_term['name'] {obs_name} is not a valid observation term.")
            
            # 记录每个观测项的信息：名字、开始序号、维度
            self.policy_obs_terms.append({
                'name': obs_name,
                'start_idx': order_index,
            'dim': obs_dim[0],
            })
            if obs_name == "height_scan":
                # 高度扫描的维度需要特殊处理
                self.height_scan_sensor_name = env.env.env.observation_manager.cfg.policy.height_scan.params['sensor_cfg'].name
                height_scan_size_x = env.env.env.scene.sensors[self.height_scan_sensor_name].cfg.pattern_cfg.size[0]
                height_scan_size_y = env.env.env.scene.sensors[self.height_scan_sensor_name].cfg.pattern_cfg.size[1]
                resolution = env.env.env.scene.sensors[self.height_scan_sensor_name].cfg.pattern_cfg.resolution
                self.height_scan_x_num = math.ceil((float(height_scan_size_x) + 1.0e-9) / float(resolution))
                self.height_scan_y_num = math.ceil((float(height_scan_size_y) + 1.0e-9) / float(resolution))

            # 更新 order_index 为下一个观测项做准备
            order_index += obs_dim[0]

    @torch.no_grad()
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        obs: TensorDict | None = None,
        actions: torch.Tensor | None = None,
    ):
        # observations
        if obs is not None:
            batch_size = obs.batch_size[0]
            # since we have 2 different symmetries, we need to augment the batch size by 2
            obs_aug = obs.repeat(2)
            # policy observation group
            # -- original
            obs_aug["policy"][:batch_size] = obs["policy"][:]
            # -- left-right
            obs_aug["policy"][batch_size : 2 * batch_size] = self._transform_policy_obs_left_right(obs["policy"])
        else:
            obs_aug = None

        # actions
        if actions is not None:
            # if isinstance(actions, dict):
            #     actions = actions["action"]
            batch_size = actions.shape[0]
            # since we have 2 different symmetries, we need to augment the batch size by 2
            actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
            # -- original
            actions_aug[:batch_size] = actions[:]
            # -- left-right
            actions_aug[batch_size : 2 * batch_size] = self._transform_actions_left_right(actions)
        else:
            actions_aug = None

        return obs_aug, actions_aug
    
    def _transform_policy_obs_left_right(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply a left-right symmetry transformation to the observation tensor.

        Args:
            obs: The observation tensor to be transformed.

        Returns:
            The transformed observation tensor with left-right symmetry applied.
        """
        # copy observation tensor
        obs = obs.clone()
        device = obs.device
        
        # 根据policy_obs_terms逐项进行变换
        for obs_info in self.policy_obs_terms:
            name = obs_info['name']
            start_idx = obs_info['start_idx'] 
            dim = obs_info['dim']
            end_idx = start_idx + dim
            
            if "ang_vel" in name:
                # 基础角速度: [wx, wy, wz] -> [-wx, wy, -wz]
                obs[:, start_idx:end_idx] *= torch.tensor([-1, 1, -1], device=device)
                
            elif "gravity" in name:
                # 投影重力: [gx, gy, gz] -> [gx, -gy, gz]
                obs[:, start_idx:end_idx] *= torch.tensor([1, -1, 1], device=device)
                
            elif name == "velocity_commands":
                # 速度命令: [vx, vy, wz] -> [vx, -vy, -wz]  
                obs[:, start_idx:end_idx] *= torch.tensor([1, -1, -1], device=device)
                
            elif name in ["joint_pos", "joint_vel", "actions"]:
                # 关节位置、速度、动作需要交换左右并应用系数
                obs[:, start_idx:end_idx] = self._switch_joints_left_right(obs[:, start_idx:end_idx])
                
            elif name == "gait_commands":
                # 步态命令需要特殊处理
                if dim == 7:  # [freq, phase_offset_left, phase_offset_right, ...]
                    obs[:, start_idx:end_idx] *= torch.tensor([1, -1, 1, 1, 1, 1, 1], device=device)
                    # 交换左右相位偏移
                    obs[:, [start_idx+3, start_idx+4, start_idx+5, start_idx+6]] = obs[:, [start_idx+5, start_idx+6, start_idx+3, start_idx+4]]
                else:
                    raise ValueError(f"Unsupported gait_commands dimension: {dim}")
            elif name == "gait_commands_quadruped":
                if dim == 13:
                    # 应用系数变换: [1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device=device)
                    # 交换四足的相位信息: FL<->FR, RL<->RR (索引5-12对应8个值)
                    # [5,6,7,8,9,10,11,12] -> [7,8,5,6,11,12,9,10]
                    temp_indices = [start_idx+i for i in [5, 6, 7, 8, 9, 10, 11, 12]]
                    swap_indices = [start_idx+i for i in [7, 8, 5, 6, 11, 12, 9, 10]]
                    obs[:, temp_indices] = obs[:, swap_indices]
                else:
                    raise ValueError(f"Unsupported gait_commands_quadruped dimension: {dim}")
                        
            elif name == "height_scan":
                obs[:, start_idx:end_idx] = (
                    obs[:, start_idx:end_idx].view(-1, self.height_scan_x_num, self.height_scan_y_num).flip(dims=[1]).view(-1, self.height_scan_x_num * self.height_scan_y_num)
                )
        
        return obs

    def _transform_actions_left_right(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply left-right symmetry transformation to actions.
        
        Args:
            actions: The actions tensor to be transformed.
            
        Returns:
            The transformed actions tensor with left-right symmetry applied.
        """
        actions = actions.clone()
        actions[:] = self._switch_joints_left_right(actions[:])
        return actions

    def _switch_joints_left_right(self, joint_data: torch.Tensor) -> torch.Tensor:
        """通用的关节左右对称变换函数
        
        This function performs left-right symmetry transformation on joint data by:
        1. Swapping left and right joint positions
        2. Applying sign flips to specific joints (e.g., roll and yaw joints)
        
        Args:
            joint_data: 关节数据张量 [batch_size, joint_dim]
            
        Returns:
            变换后的关节数据张量
            
        Raises:
            ValueError: 如果关节数据维度与配置不匹配
        """
        config = self._get_joint_transform_config(self.robot_name)
        expected_joint_count = config["joint_count"]
        
        # 验证输入数据的维度
        if joint_data.shape[-1] != expected_joint_count:
            raise ValueError(
                f"Joint data dimension ({joint_data.shape[-1]}) does not match "
                f"expected joint count ({expected_joint_count}) for {self.robot_name}"
            )
        
        joint_data_switched = torch.zeros_like(joint_data)
        
        left_indices = config["left_indices"]
        right_indices = config["right_indices"]
        flip_indices = config["flip_indices"]
        
        # 交换左右关节：左 <-- 右，右 <-- 左
        joint_data_switched[..., left_indices] = joint_data[..., right_indices]
        joint_data_switched[..., right_indices] = joint_data[..., left_indices]
        
        # 对特定关节取反（通常是roll和yaw关节）
        joint_data_switched[..., flip_indices] *= -1.0
        
        return joint_data_switched
    
    def _get_joint_transform_config(self, robot_name: str) -> dict:
        """获取机器人关节变换配置
        
        Returns:
            dict: 包含关节变换配置的字典
                - joint_count: 关节总数
                - left_indices: 左侧关节索引列表
                - right_indices: 右侧关节索引列表  
                - flip_indices: 需要取反的关节索引列表
        """
        joint_configs = {
            "G1_12dof": {
                "joint_count": 12,
                "left_indices": [0, 1, 2, 3, 4, 5],      # 左腿关节索引
                "right_indices": [6, 7, 8, 9, 10, 11],   # 右腿关节索引
                "flip_indices": [1, 2, 5, 7, 8, 11],     # hip_roll, hip_yaw, ankle_roll 需要取反
            },
            "G1_29dof": {
                "joint_count": 29,
                # 左侧：左腿(0-5) + 左臂(15-21) = 13个关节
                "left_indices": [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21],
                # 右侧：右腿(6-11) + 右臂(22-28) = 13个关节
                "right_indices": [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 28],
                # 需要取反的关节：
                # - 腿部: hip_roll(1,7), hip_yaw(2,8), ankle_roll(5,11)
                # - 腰部: waist_yaw(12), waist_roll(13)
                # - 手臂: shoulder_roll(16,23), shoulder_yaw(17,24), wrist_roll(19,26), wrist_yaw(21,28)
                # "flip_indices": [1, 2, 5, 7, 8, 11, 13, 16, 17, 19, 21, 23, 24, 26, 28],
                "flip_indices": [1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28],
            },
            "S3_22dof": {
                "joint_count": 12,  # per-side joint count used in switching functions (6 left + 6 right = 12)
                # In S3_symmetry.py the switching expects a 12-dim block: left(0-5) and right(6-11)
                "left_indices": [0, 1, 2, 3, 4, 5],
                "right_indices": [6, 7, 8, 9, 10, 11],
                # Flip the sign of hip_roll, hip_yaw, ankle_roll for both sides (indices after switching)
                # In S3_symmetry._switch_S3_joints_left_right the flipped indices are [0,1,5,6,7,11] after swapping
                # To express as positions in the joint_data before switching (matching left/right layout),
                # we list the indices that should be negated after the swap: these correspond to the same set
                # of joint names on both sides.
                "flip_indices": [0, 1, 5, 6, 7, 11],
            },
            "Go2": {
                "joint_count": 12,
                # Go2关节顺序: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
                # 左侧：FL(0-2) + RL(6-8) = 6个关节
                "left_indices": [0, 1, 2, 6, 7, 8],
                # 右侧：FR(3-5) + RR(9-11) = 6个关节  
                "right_indices": [3, 4, 5, 9, 10, 11],
                # Go2的hip关节是roll轴，左右对称时需要取反
                # hip关节索引: FL_hip(0), FR_hip(3), RL_hip(6), RR_hip(9)
                "flip_indices": [0, 3, 6, 9],
            },
            
        }
        
        if robot_name not in joint_configs:
            raise ValueError(f"Unsupported robot: {robot_name}. Supported robots: {list(joint_configs.keys())}")
        
        config = joint_configs[robot_name]
        
        # 验证配置的一致性
        if len(config["left_indices"]) != len(config["right_indices"]):
            raise ValueError(f"Left and right joint counts must be equal for {robot_name}")
        
        return config



@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """
    便利函数：创建对称状态计算器并执行变换
    
    Args:
        env: 环境实例
        obs: 观测张量
        actions: 动作张量
        
    Returns:
        tuple: (变换后的观测, 变换后的动作)
    """
    # 创建计算器实例（可以考虑缓存以提高性能）
    if not hasattr(compute_symmetric_states, '_cache'):
        compute_symmetric_states._cache = {}
    
    # 使用环境配置作为缓存键
    env_key = str(env.cfg.scene.robot.spawn.usd_path)
    if env_key not in compute_symmetric_states._cache:
        compute_symmetric_states._cache[env_key] = Compute_symmetric_states(env)
    
    computer = compute_symmetric_states._cache[env_key]
    return computer(env, obs, actions)
    
