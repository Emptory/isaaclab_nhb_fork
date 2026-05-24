from __future__ import annotations

import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerBase, ManagerTermBase, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg 
from isaaclab.assets import Articulation, DeformableObject, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs import ManagerBasedEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    command_term_name: str = "base_velocity",
    step: float = 0.1,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_term_name)
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-step, step], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)

def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
    command_term_name: str = "base_velocity",
    step: float = 0.1,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_term_name)
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-step, step], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)

class reward_weight_tracking_levels(ManagerTermBase):
    """根据速度跟踪奖励值逐渐增大某个奖励函数的权重"""
    
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.init_flag = False
        self.env = env
        self.raw_weight = None
        self.current_ratio = 0.0

    def param_init(self, reward_term_name: str, init_ratio: float):
        """初始化：保存原始权重并设置初始比例"""
        self.raw_weight = self.env.reward_manager.get_term_cfg(reward_term_name).weight
        self.current_ratio = init_ratio
        # 设置初始权重
        self.env.reward_manager.get_term_cfg(reward_term_name).weight = self.raw_weight * init_ratio
        self.init_flag = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str,
        tracking_reward_name: str = "track_lin_vel_xy_exp",
        step: float = 0.02,
        reward_threshold: float = 0.8,
        init_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        根据速度跟踪奖励值逐渐增大目标奖励函数的权重
        
        Args:
            env: 环境实例
            env_ids: 环境ID列表
            reward_term_name: 要调整权重的目标奖励函数名称
            tracking_reward_name: 参考的速度跟踪奖励名称
            step: 每次权重比例增加的步长
            reward_threshold: 触发权重增加的奖励阈值(相对于权重的比例)
            init_ratio: 初始权重比例 [0, 1]，0表示完全关闭，1表示完全启用
            
        Returns:
            torch.Tensor: 当前的奖励权重
        """
        # 初始化
        if not self.init_flag:
            self.param_init(reward_term_name, init_ratio)
        
        # 获取速度跟踪奖励的表现
        tracking_reward_term = env.reward_manager.get_term_cfg(tracking_reward_name)
        tracking_reward = torch.mean(
            env.reward_manager._episode_sums[tracking_reward_name][env_ids]
        ) / env.max_episode_length_s
        
        # 每隔一定步数检查一次是否需要增大权重
        if env.common_step_counter % (env.max_episode_length // 2) == 0:
            # 如果速度跟踪奖励超过阈值，逐步增大目标奖励的权重
            if tracking_reward > tracking_reward_term.weight * reward_threshold:
                self.current_ratio = min(self.current_ratio + step, 1.0)
        
        # 更新目标奖励函数的权重
        self.env.reward_manager.get_term_cfg(reward_term_name).weight = self.raw_weight * self.current_ratio
        
        return torch.tensor(
            self.env.reward_manager.get_term_cfg(reward_term_name).weight, 
            device=env.device
        )

class reward_weight_episodes_levels(ManagerTermBase):
    """根据episode长度对奖励函数的权重进行调整"""
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.init_flag = False
        self.env = env


    def param_init(self, reward_term_name: str, init_ratio: float):
        # 获取奖励函数的初始权重
        self.raw_weight = self.env.reward_manager.get_term_cfg(reward_term_name).weight
        self.init_flag = True
        self.ratio = init_ratio

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str,
        step: float = 0.01,
        init_ratio: float = 0.0,
    )-> torch.Tensor:
        """
        reward_term_name: 被调整的奖励函数的名称
        rate: 下一次episode的权重更新比例,
        当前的权重比例= rate * 上一次的比例 + (1 - rate) * 当前episode的长度占最大episode长度的比例
        """
        # 初始化
        if not self.init_flag:
            self.param_init(reward_term_name,init_ratio)
        
        # 计算当前episode占最大episode的比例
        if env.common_step_counter % (env.max_episode_length / 2) == 0: # 每500个step判断一次
            if torch.mean(env.episode_length_buf.float()) > env.max_episode_length * 0.35:
                self.ratio = min(self.ratio + step, 1.0)
        # 更新奖励函数的权重
        self.env.reward_manager.get_term_cfg(reward_term_name).weight = self.raw_weight * self.ratio
        return self.env.reward_manager.get_term_cfg(reward_term_name).weight

def lin_vel_cmd_std_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    step: float = 0.1,
    min_std: float = 0.1
) -> torch.Tensor:
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            env.reward_manager.get_term_cfg(reward_term_name).params["std"] = max(
                env.reward_manager.get_term_cfg(reward_term_name).params["std"] - step,
                min_std,
            )

    return torch.tensor(env.reward_manager.get_term_cfg(reward_term_name).params["std"], device=env.device)

def random_push_x_vel_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    step: float = 0.05,
    max_vel: float = 1.0
) -> torch.Tensor:
    """根据线速度跟踪奖励函数的表现调整随机推力的强度,针对x方向

    Args:
        env (ManagerBasedRLEnv): 环境实例
        env_ids (Sequence[int]): 环境ID列表
        reward_term_name (str, optional): 线速度跟踪奖励. Defaults to "track_lin_vel_xy_exp".
        step (float, optional): 每次调整的步长. Defaults to 0.05.
        max_vel (float, optional): 最大速度. Defaults to 1.0.

    Returns:
        torch.Tensor: 当前的随机推力强度
    """
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            x_range = list(env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["x"])
            x_range[1] = min(x_range[1] + step, max_vel)
            x_range[0] = max(x_range[0] - step, -max_vel)
            env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["x"] = tuple(x_range)

    return torch.tensor(env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["x"][1], device=env.device)

def random_push_y_vel_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    step: float = 0.05,
    max_vel: float = 1.0
) -> torch.Tensor:
    """根据线速度跟踪奖励函数的表现调整随机推力的强度,针对y方向

    Args:
        env (ManagerBasedRLEnv): 环境实例
        env_ids (Sequence[int]): 环境ID列表
        reward_term_name (str, optional): 线速度跟踪奖励. Defaults to "track_lin_vel_xy_exp".
        step (float, optional): 每次调整的步长. Defaults to 0.05.
        max_vel (float, optional): 最大速度. Defaults to 1.0.

    Returns:
        torch.Tensor: 当前的随机推力强度
    """
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            y_range = list(env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["y"])
            y_range[1] = min(y_range[1] + step, max_vel)
            y_range[0] = max(y_range[0] - step, -max_vel)
            env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["y"] = tuple(y_range)

    return torch.tensor(env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["y"][1], device=env.device)

def random_push_z_vel_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    step: float = 0.05,
    max_vel: float = 1.0
) -> torch.Tensor:
    """根据线速度跟踪奖励函数的表现调整随机推力的强度,针对z方向

    Args:
        env (ManagerBasedRLEnv): 环境实例
        env_ids (Sequence[int]): 环境ID列表
        reward_term_name (str, optional): 线速度跟踪奖励. Defaults to "track_lin_vel_xy_exp".
        step (float, optional): 每次调整的步长. Defaults to 0.05.
        max_vel (float, optional): 最大速度. Defaults to 1.0.

    Returns:
        torch.Tensor: 当前的随机推力强度
    """
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            z_range = list(env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["z"])
            z_range[1] = min(z_range[1] + step, max_vel)
            z_range[0] = max(z_range[0] - step, -max_vel)
            env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["z"] = tuple(z_range)

    return torch.tensor(env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["z"][1], device=env.device)
                          
def random_joints_target_position_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    step: float = 0.05,
    high_clip: float = 1.0,
    reward_term_name: str = "track_lin_vel_xy_exp",
    reward_threshold: float = 0.8,
):
    """课程，随着速度跟踪奖励的上升，逐步增加随机化关节目标位置的范围

    Args:
        step (float, optional): 每次范围增加的步进值. Defaults to 0.05.
        high_clip (float, optional): 关节随机化的最大限幅,最高只能是1.0. Defaults to 1.0.
        reward_term_name (str, optional): 速度跟踪奖励的名称. Defaults to "track_lin_vel_xy_exp".
        reward_threshold (float, optional): 触发范围增加的奖励阈值(相对于权重的比例). Defaults to 0.8.
    """

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * reward_threshold:
            position_range = list(env.event_manager.get_term_cfg("random_joints_target_position").params["position_range"])
            position_range[1] = min(position_range[1] + step, high_clip)
            env.event_manager.get_term_cfg("random_joints_target_position").params["position_range"] = tuple(position_range)
    return torch.tensor(env.event_manager.get_term_cfg("random_joints_target_position").params["position_range"][1], device=env.device)

class joint_pos_limit_curriculum(ManagerTermBase):
    """根据速度跟踪奖励逐步放开特定关节的位置限幅
    
    在训练初期缩小腰部和手部关节的位置限幅范围,随着速度跟踪奖励提升逐步恢复到原始限幅值
    """
    
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.init_flag = False
        self.original_limits = None  # 保存原始的关节位置限幅 (num_envs, num_joints, 2)
        self.joint_indices = None  # 要控制的关节索引
        self.current_ratio = 0.0  # 当前恢复比例 [0, 1]
        self.asset_cfg = None
        
    def param_init(self, asset_cfg: SceneEntityCfg, init_ratio: float):
        """初始化时保存原始的关节位置限幅并缩小范围"""
        from isaaclab.assets import Articulation
        
        asset: Articulation = self.env.scene[asset_cfg.name]
        
        # 从 asset_cfg 中获取关节名称列表（支持正则表达式）
        joint_name_patterns = asset_cfg.joint_names if asset_cfg.joint_names else []
        
        # 展开正则表达式匹配所有关节
        all_joint_names = asset.data.joint_names
        expanded_joints = []
        for pattern in joint_name_patterns:
            for i, joint_name in enumerate(all_joint_names):
                if re.match(pattern, joint_name):
                    if i not in expanded_joints:
                        expanded_joints.append(i)
        
        self.joint_indices = torch.tensor(expanded_joints, device=asset.device, dtype=torch.long)
        
        # 保存原始的关节位置限幅
        self.original_limits = asset.data.default_joint_pos_limits[:, self.joint_indices, :].clone()
        
        # 初始化恢复比例
        self.current_ratio = init_ratio
        
        # 计算初始的缩小限幅 (围绕默认位置缩小)
        default_pos = asset.data.default_joint_pos[:, self.joint_indices]
        
        # 缩小限幅: new_limit = default_pos + (original_limit - default_pos) * ratio
        new_limits = self.original_limits.clone()
        new_limits[:, :, 0] = default_pos + (self.original_limits[:, :, 0] - default_pos) * init_ratio
        new_limits[:, :, 1] = default_pos + (self.original_limits[:, :, 1] - default_pos) * init_ratio
        
        # 写入物理仿真
        asset.write_joint_position_limit_to_sim(
            new_limits, 
            joint_ids=self.joint_indices, 
            env_ids=None,
            warn_limit_violation=False
        )
        
        self.init_flag = True
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg,
        reward_term_name: str = "track_lin_vel_xy_exp",
        step: float = 0.02,
        reward_threshold: float = 0.7,
        init_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        根据速度跟踪奖励逐步恢复关节的位置限幅
        
        Args:
            env: 环境实例
            env_ids: 环境ID列表
            asset_cfg: 资产配置，包含 joint_names 列表（支持正则表达式）
            reward_term_name: 参考的速度跟踪奖励名称
            step: 每次恢复的步长(比例)
            reward_threshold: 触发恢复的奖励阈值(相对于权重的比例)
            init_ratio: 初始限幅比例 [0, 1], 0表示完全锁住, 1表示完全放开
        
        Returns:
            当前的恢复比例 [0, 1]
        """
        from isaaclab.assets import Articulation
        
        # 初始化
        if not self.init_flag:
            self.asset_cfg = asset_cfg
            self.param_init(asset_cfg, init_ratio)
        
        asset: Articulation = self.env.scene[asset_cfg.name]
        
        # 获取速度跟踪奖励
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
        
        # 每隔一定步数检查一次是否需要提升限幅
        if env.common_step_counter % (env.max_episode_length // 2) == 0:
            # 如果奖励超过阈值,逐步恢复关节限幅
            if reward > reward_term.weight * reward_threshold:
                old_ratio = self.current_ratio
                self.current_ratio = min(self.current_ratio + step, 1.0)
                
                # 只在比例变化时更新限幅
                if self.current_ratio > old_ratio:
                    # 计算新的限幅
                    default_pos = asset.data.default_joint_pos[:, self.joint_indices]
                    new_limits = self.original_limits.clone()
                    new_limits[:, :, 0] = default_pos + (self.original_limits[:, :, 0] - default_pos) * self.current_ratio
                    new_limits[:, :, 1] = default_pos + (self.original_limits[:, :, 1] - default_pos) * self.current_ratio
                    
                    # 写入物理仿真
                    asset.write_joint_position_limit_to_sim(
                        new_limits,
                        joint_ids=self.joint_indices,
                        env_ids=None,
                        warn_limit_violation=False
                    )
        
        return torch.tensor(self.current_ratio, device=env.device)