"""
自己编写的奖励函数
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING,Sequence, List, Dict, Optional

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import yaw_quat, quat_apply_inverse
from isaaclab.managers import ManagerBase, ManagerTermBase
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_rotate, quat_apply, quat_error_magnitude
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera 
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz
from .commands import MotionCommand
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


###################################### 下面是任务奖励函数 ########################################


########################### 机身高度跟踪衍生奖励函数 ###################################

def base_height_exp(
        env: ManagerBasedRLEnv,
        target_height: float,
        std: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
    """使用指数核奖励机器人保持目标高度
    
    Args:
        target_height: 目标高度（米）
        std: 高斯核标准差，控制奖励衰减速度，越小对误差越敏感
        asset_cfg: 机器人资产配置
        sensor_cfg: 高度传感器配置（用于地形高度补偿）

    Returns:
        奖励值，范围[0, 1]，权重应为正数
    """
    # 获取机器人
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算调整后的目标高度
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # 使用传感器数据调整目标高度
        terrain_height = torch.clip(torch.mean(sensor.data.ray_hits_w[..., 2], dim=1), -10.0, 10.0)
        adjusted_target_height = target_height + terrain_height
    else:
        # 平坦地形直接使用目标高度
        adjusted_target_height = target_height
    
    # 计算高度误差
    height_error = asset.data.root_pos_w[:, 2] - adjusted_target_height
    
    # 使用指数核计算奖励
    return torch.exp(-torch.square(height_error) / std**2)

########################### 速度命令跟踪衍生奖励函数 ###################################

def track_lin_vel_xy_yaw_frame_expabs(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    """
    使用abs高斯核给出xy方向的速度跟踪奖励
    权重应为正
    """

    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.abs(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )

    return torch.exp(-lin_vel_error / std**2)

def track_lin_vel_xyz_yaw_frame_abs(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    """
    使用abs给出xyz方向的速度跟踪奖励
    权重应为正
    """

    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.abs(env.command_manager.get_command(command_name)[:, :3] - vel_yaw[:, :3]), dim=1
    )

    return torch.clip(-5 * lin_vel_error + 1, -1, 1)

def track_ang_vel_z_world_expabs(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """
    使用abs高斯核给出z方向角速度跟踪奖励
    权重应为正
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.abs(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

########################### 危险落足惩罚 ###################################

class TerrainEdgePenalty(ManagerTermBase):
    """
    基于边缘检测的地形边缘惩罚

    """
    
    _env: ManagerBasedRLEnv
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # 初始化传感器
        self.raycaster_sensor_cfgs = cfg.params["raycaster_sensor_cfgs"]
        # self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.raycaster_sensors = [env.scene.sensors[cfg.name] for cfg in self.raycaster_sensor_cfgs]
        pattern_cfg = self.raycaster_sensors[0].cfg.pattern_cfg

        # 计算RayCaster的长宽
        resolution = float(pattern_cfg.resolution)
        size_x, size_y = float(pattern_cfg.size[0]), float(pattern_cfg.size[1])
        self.grid_rows = math.ceil((size_x + 1.0e-9) / resolution)
        self.grid_cols = math.ceil((size_y + 1.0e-9) / resolution)
        self.num_rays = self.grid_rows * self.grid_cols
        self.max_edge_dist_x = size_x / 2.0
        self.max_edge_dist_y = size_y / 2.0

        # 传感器参数检查
        if not hasattr(pattern_cfg, "resolution") or not hasattr(pattern_cfg, "size"):
            raise ValueError("TerrainEdgePenaltyOld 仅支持 GridPatternCfg 类型的足底 RayCaster。")
        if getattr(pattern_cfg, "ordering", "yx") != "yx":
            raise ValueError("TerrainEdgePenaltyOld 当前要求 RayCaster GridPatternCfg.ordering == 'yx'。")

        for sensor in self.raycaster_sensors[1:]:
            sensor_pattern_cfg = sensor.cfg.pattern_cfg
            sensor_rows = math.ceil(
                (float(sensor_pattern_cfg.size[0]) + 1.0e-9) / float(sensor_pattern_cfg.resolution)
            )
            sensor_cols = math.ceil(
                (float(sensor_pattern_cfg.size[1]) + 1.0e-9) / float(sensor_pattern_cfg.resolution)
            )
            if sensor_rows != self.grid_rows or sensor_cols != self.grid_cols:
                raise ValueError(
                    "TerrainEdgePenaltyOld 要求所有足底 RayCaster 使用相同的 GridPatternCfg 尺寸与分辨率。"
                )
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        raycaster_sensor_cfgs: Sequence[SceneEntityCfg],
        d_sens: float = 0.05,
        dist_offset: float = 0.035,
        edge_threshold: float = 0.03,
    ) -> torch.Tensor:
        """
            1. 使用每个足底 RayCaster 生成局部高程图；
            2. 通过高程图在 x/y 方向的梯度检测足底附近是否存在地形边缘；
            3. 估计足端到边缘中心的方向，并结合当前速度命令，判断机器人是否在朝危险边缘运动；
            4. 当边缘距离足端较近时，给出更强的惩罚，避免踩空、踩楼梯边缘或靠近危险落足区域。
            
            env: 当前 ManagerBasedRLEnv 环境实例。
            raycaster_sensor_cfgs: 足底 RayCaster 的配置列表，保留在函数签名中主要是为了兼容 RewardTerm 的调用方式。
            d_sens: 距离敏感范围，值越大表示离边缘更远时仍然会保留惩罚权重。
            dist_offset: 距离偏移量，可理解为“安全距离基线”；当足端离边缘小于该基线附近时惩罚更明显。
            edge_threshold: 边缘检测阈值；高程图梯度幅值超过该值的位置会被认为是边缘点。
        """
        
        # 获取RayCaster数据并转移到局部坐标系
        ray_hits_w = torch.stack([s.data.ray_hits_w for s in self.raycaster_sensors], dim=1)  # [num_envs, num_feet, num_rays, 3]
        ray_origins_w = torch.stack([s.data.pos_w for s in self.raycaster_sensors], dim=1)  # [num_envs, num_feet, 3]
        sensor_pos_expanded = ray_origins_w.unsqueeze(2) 
        ray_vectors = ray_hits_w - sensor_pos_expanded # [num_envs, num_feet, num_rays, 3]
        ray_vectors = torch.clip(ray_vectors, min=-10.0, max=10.0)
        
        # ========== 1. 边缘检测 ==========
        # 获取RayCaster的形状信息
        num_envs, num_feet, num_rays, _ = ray_vectors.shape
        
        # 提取z值（高度图）：[B, F, rows, cols]
        height_map = ray_vectors[..., 2].reshape(num_envs, num_feet, self.grid_rows, self.grid_cols)
         
        # 计算x和y方向的梯度
        grad_x = height_map[..., 1:, :] - height_map[..., :-1, :] # 梯度x: [B, F, grid_size-1, grid_size]，前减后
        grad_y = height_map[..., :, 1:] - height_map[..., :, :-1] # [B, F, grid_size, grid_size-1]，左减右

        # 排除不明显的梯度
        grad_x[torch.abs(grad_x) < 0.02] = 0.0
        grad_y[torch.abs(grad_y) < 0.02] = 0.0

        # 由于差分少了一维，所以先对少的一维末尾补零
        grad_x_padded = torch.nn.functional.pad(grad_x, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant', value=0.0)
        grad_y_padded = torch.nn.functional.pad(grad_y, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant', value=0.0)

        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(grad_x_padded**2 + grad_y_padded**2)  # [B, F, grid_size, grid_size]
        
        # 边缘掩码：梯度超过阈值的位置
        edge_mask = gradient_magnitude > edge_threshold  # [B, F, grid_size, grid_size]
        
        # 重塑回原始射线形状
        edge_mask = edge_mask.reshape(num_envs, num_feet, num_rays)  # [B, F, R]
        
        # ========== 2. 计算边缘中心 ==========
        # 首先统计每个足端有多少边缘点 [B, F]
        edge_count = edge_mask.sum(dim=-1)  
        
        # 如果没有边缘点，边缘中心设为足端中心（避免零除）
        has_edge = edge_count > 0
        
        # 计算边缘点的平均位置（世界坐标）
        # 使用mask进行加权平均
        edge_mask_expanded = edge_mask.unsqueeze(-1).float()  # [B, F, R, 1]
        edge_points_weighted = ray_vectors * edge_mask_expanded  # [B, F, R, 3]
        edge_center = edge_points_weighted.sum(dim=-2) / (edge_count.unsqueeze(-1) + 1e-6)  # [B, F, 3]
        
        # 如果没有边缘，边缘中心等于足端中心（向量将为零）
        edge_center = torch.where(
            has_edge.unsqueeze(-1),
            edge_center,
            torch.zeros_like(ray_origins_w)
        )  # [B, F, 3]
        
        # ========== 3. 计算从足端到边缘中心的向量 ==========

        foot_to_edge_vector_xy = edge_center[..., :2]

        # 顺序与符号修正、归一化
        foot_to_edge_vector_xy = foot_to_edge_vector_xy[..., [1, 0]] # 因为配置的order是"yx"，所以这
        foot_to_edge_vector_xy[..., 1] *= -1
        foot_to_edge_vector_xy[..., 0] /= max(self.max_edge_dist_x, 1e-6)
        foot_to_edge_vector_xy[..., 1] /= max(self.max_edge_dist_y, 1e-6)

        # 获取速度命令向量
        now_vel_cmd = env.command_manager.get_command("base_velocity")[:, :2].unsqueeze(-2)
        now_vel_cmd_norm = now_vel_cmd / (torch.norm(now_vel_cmd, dim=-1, keepdim=True) + 1e-6)
        # ========== 5. 点积判断 ==========
        # 计算点积：足端→边缘方向 与 速度方向 的点积
        dot_product = torch.sum(foot_to_edge_vector_xy * now_vel_cmd_norm, dim=-1)  # [B, F]

        # 符号项
        mean_grad_x = torch.mean(grad_x, dim=[-1, -2])  # [B, F]
        mean_grad_y = torch.mean(grad_y, dim=[-1, -2])  # [B, F]
        g_dot_v = mean_grad_x * now_vel_cmd[..., 0] + mean_grad_y * now_vel_cmd[..., 1]  # [B, F]
        sign = -torch.sign(g_dot_v)

        # 边缘惩罚项
        p_edge = torch.minimum(dot_product * sign, torch.zeros_like(dot_product))

        # ========== 6. 距离权重 ==========
        # 计算射线最小z距离（距离地面的距离）
        d_min, _ = torch.min(torch.abs(ray_vectors[..., 2]), dim=-1, keepdim=True)  # [B, F, 1]
        p_dist = torch.relu(1.0 - (d_min.squeeze(-1) - dist_offset) / d_sens)  # [B, F]
        
        return torch.mean(p_edge * p_dist, dim=1)
    
    def _format_height_map_for_terminal(height_map_0: torch.Tensor) -> str:
        """Format ``height_map[0]`` for terminal display.

        Display convention:
        - terminal 右下角为原点
        - 往上是 x 增大
        - 往左是 y 增大
        """

        display_map = torch.flip(height_map_0.detach(), dims=(-2, -1)).cpu().numpy()

        lines = []
        for foot_id, foot_map in enumerate(display_map):
            lines.append(f"[height_map[0] | foot={foot_id}] x+ ↑, y+ ←, origin at bottom-right")
            lines.append(np.array2string(foot_map, precision=3, suppress_small=False))
        return "\n".join(lines)
    
class FeetCollisionPenalty(ManagerTermBase):
    """基于射线检测的足端碰撞惩罚 (完全向量化优化版)"""
    
    _env: ManagerBasedRLEnv
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 检查传感器维度
        self.num_sensors = len(cfg.params["raycaster_sensor_cfgs"])
        self.num_feet = len(cfg.params["asset_cfg"].body_names)
        if self.num_sensors != self.num_feet * 2:
            raise ValueError(
                f"奖励函数[feet_collision_penalty]错误, 传感器数量({self.num_sensors})与足端数量×2({self.num_feet * 2})不匹配！"
                f"每个足端需要两个传感器（下层、上层）"
            )
        
        # 初始化配置参数
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        sensors = [env.scene.sensors[cfg.name] for cfg in cfg.params["raycaster_sensor_cfgs"]]
        self.d_safe = cfg.params["d_safe"]
        self.slope_threshold = cfg.params["slope_threshold"]
        self.cone_angle = cfg.params["cone_angle"]

        # 预计算锥形余弦值
        self.cone_cos = math.cos(math.radians(self.cone_angle))

        # 速度滤波参数
        self.vel_buffer_size = cfg.params["vel_buffer_size"]
        
        # 分离下层和上层传感器
        self.lower_sensors = sensors[0::2] # [起始索引:结束索引:步长]
        self.upper_sensors = sensors[1::2]
        
        # 初始化足端速度历史缓冲区 [buffer_size, num_envs, num_feet, 2]
        self.foot_vel_buffer = torch.zeros(
            self.vel_buffer_size, self._env.num_envs, self.num_feet, 2,
            dtype=torch.float32, device=self._env.device
        )
        
        # 初始化左脚速度矢量可视化器
        self.arrow_visualizer = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/LeftFootVelocityArrow"))
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        raycaster_sensor_cfgs: Sequence[SceneEntityCfg],
        asset_cfg: SceneEntityCfg,
        d_safe: float,
        slope_threshold: float,
        cone_angle: float,
        vel_buffer_size: int,
    ) -> torch.Tensor:
        """
        计算基于射线检测的足端碰撞惩罚1.
        
        Args:
            env: 环境实例
            raycaster_sensor_cfgs: 传感器配置列表 [L_low, L_up, R_low, R_up, ...]
            asset_cfg: 足端资产配置
            
        Returns:
            torch.Tensor: [num_envs] 平均碰撞惩罚 (负值)
        """
        
        # 获取足端速度 [B, F, 3] -> [B, F, 2] (只取 xy)
        foot_vel_xy_w = self.asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        # 更新足端速度历史缓冲区
        self.foot_vel_buffer = torch.roll(self.foot_vel_buffer, shifts=1, dims=0)
        self.foot_vel_buffer[0] = foot_vel_xy_w.clone()
        # 计算均值滤波后的足端速度
        foot_vel_xy_filtered = torch.mean(self.foot_vel_buffer, dim=0)  # [B, F, 2]
        
        # 获取所有传感器位置与击中点 -> Stack 为 [B, F, R, 3] B: num_envs, F: num_feet, R: num_rays
        lower_pos_w = torch.stack([s.data.pos_w for s in self.lower_sensors], dim=1) # [B, F, 3]
        lower_hits_w = torch.stack([s.data.ray_hits_w for s in self.lower_sensors], dim=1)
        upper_hits_w = torch.stack([s.data.ray_hits_w for s in self.upper_sensors], dim=1)
        # 直接clip处理inf和nan（当射线未击中时）
        # lower_hits_w = torch.clamp(lower_hits_w, min=-99999.0, max=99999.0)
        # upper_hits_w = torch.clamp(upper_hits_w, min=-99999.0, max=99999.0)
        
        # 3. 计算射线方向与距离 (下层)
        # 射线向量: hit - origin -> [B, F, R, 3] 需要将 origin 扩展为 [B, F, 1, 3] 以便广播
        ray_vecs = lower_hits_w - lower_pos_w.unsqueeze(2)
        # 归一化射线方向 (XY平面) -> [B, F, R, 2]
        ray_dists_xy = torch.norm(ray_vecs[..., :2], dim=-1)
        ray_dirs_xy = ray_vecs[..., :2] / (ray_dists_xy.unsqueeze(-1) + 1e-6)
        
        # 4. 锥形区域筛选 归一化足端速度 -> [B, F, 2]
        foot_vel_norm = torch.norm(foot_vel_xy_filtered, dim=-1, keepdim=True)
        foot_vel_dir = foot_vel_xy_filtered / (foot_vel_norm + 1e-6)
        
        # 计算点积 foot_vel: [B, F, 1, 2], ray_dirs: [B, F, R, 2] -> [B, F, R]
        cos_theta = torch.sum(foot_vel_dir.unsqueeze(2) * ray_dirs_xy, dim=-1)
        
        # 锥形掩码 [B, F, R]
        is_in_cone = cos_theta > self.cone_cos
        # 将锥形外的距离设为无穷大
        valid_dists = torch.where(is_in_cone, ray_dists_xy, torch.tensor(float('inf'), device=self._env.device))
        # 找到最近距离和对应的索引 -> [B, F]
        min_dist, min_indices = torch.min(valid_dists, dim=-1)
        
        # 6. 获取最近射线对应的上下层击中点 (Gathering)
        # min_indices 形状 [B, F]，需要扩展为 [B, F, 1, 3] 以配合 gather
        gather_indices = min_indices.view(min_indices.shape[0], self.num_feet, 1, 1).expand(-1, -1, 1, 3)
        # 提取最近射线的击中点 -> [B, F, 1, 3] -> squeeze -> [B, F, 3]
        closest_lower_hit = torch.gather(lower_hits_w, 2, gather_indices).squeeze(2)
        closest_upper_hit = torch.gather(upper_hits_w, 2, gather_indices).squeeze(2)
        
        # 提取最近射线的方向 -> [B, F, 2]
        # 为了准确，我们gather之前算好的 ray_dirs_xy
        gather_indices_xy = min_indices.view(min_indices.shape[0], self.num_feet, 1, 1).expand(-1, -1, 1, 2)
        closest_ray_dir_xy = torch.gather(ray_dirs_xy, 2, gather_indices_xy).squeeze(2)
        
        
        # 坡度计算与楼梯判定 
        delta_xy = torch.norm(closest_upper_hit[..., :2] - closest_lower_hit[..., :2], dim=-1)
        delta_z = torch.abs(closest_upper_hit[..., 2] - closest_lower_hit[..., 2])
        # 计算坡度 (delta_z / delta_xy)
        slope = delta_z / (delta_xy + 1e-5)
        is_stair = slope > self.slope_threshold # [B, F]
        
        # 计算惩罚项 
        # 速度投影项 ReLU(v · d)
        vel_proj = torch.sum(foot_vel_xy_filtered * closest_ray_dir_xy, dim=-1) # [B, F]
        term_vel = torch.relu(vel_proj)
        # 距离项 ReLU(1 - d / d_safe)
        term_dist = torch.relu(1.0 - min_dist / self.d_safe)
        
        # 计算最终的惩罚值
        penalty_per_foot = torch.where(
            is_stair,
            term_vel * term_dist,
            torch.zeros_like(term_vel)
        )
        mean_penalty = torch.mean(penalty_per_foot, dim=1) 

        # 可视化debug
        # self._visualize_foot_velocity(asset_cfg)
        # self._visualize_min_dist_arrow(lower_pos_w, closest_lower_hit)
        
        return mean_penalty
    
    def reset(self, env_ids = None):
        """当环境被reset的时候，清空速度缓冲区"""
        self.foot_vel_buffer[:, env_ids] *= 0

    def _visualize_min_dist_arrow(self, ray_start_pos: torch.Tensor, ray_end_pos: torch.Tensor):
        """
        可视化从射线起点到终点的箭头（min_dist对应的射线）
        
        Args:
            ray_start_pos: 射线起点坐标 [B, F, 3]，其中B是环境数，F是足端数
            ray_end_pos: 射线终点坐标 [B, F, 3]
        """
        # 展平数据：[B, F, 3] -> [B*F, 3]
        start_pos_flat = ray_start_pos.reshape(-1, 3)
        end_pos_flat = ray_end_pos.reshape(-1, 3)
        
        # 计算射线向量（从起点到终点）
        ray_vector = end_pos_flat - start_pos_flat
        
        # 计算射线方向（归一化）
        ray_length = torch.norm(ray_vector, dim=1, keepdim=True)
        ray_dir = ray_vector / (ray_length + 1e-6)
        
        # 计算偏航角（yaw）和俯仰角（pitch）
        # ray_dir格式为 [x, y, z]
        yaw_angle = torch.atan2(ray_dir[:, 1], ray_dir[:, 0])
        pitch_angle = -torch.asin(torch.clamp(ray_dir[:, 2], -1.0, 1.0))  # 负号因为pitch的定义方向
        
        # 创建四元数，先绕y轴旋转pitch，再绕z轴旋转yaw
        zeros = torch.zeros_like(yaw_angle)
        # arrow_quat = quat_from_euler_xyz(zeros, pitch_angle, yaw_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, yaw_angle)
        
        # 缩放箭头长度（基于射线长度）
        default_scale = self.arrow_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(start_pos_flat.shape[0], 1)
        arrow_scale[:, 0] *= ray_length.squeeze()  # x轴缩放对应箭头长度
        
        # 可视化箭头（使用起点位置作为箭头位置）
        self.arrow_visualizer.visualize(
            translations=start_pos_flat + 0.5 * ray_vector,
            orientations=arrow_quat,
            scales=arrow_scale
        )
    
    def _visualize_foot_velocity(self, asset_cfg: SceneEntityCfg):
        """
        可视化机器人的左脚速度矢量
        
        Args:
            asset_cfg: 足端资产配置
        """
        # 获取左脚的位置和速度（假设左脚是第一个足端，索引为0）
        foot_idx = asset_cfg.body_ids  
        foot_pos = self.asset.data.body_pos_w[:, foot_idx].view(-1, 3)
        foot_vel = self.asset.data.body_lin_vel_w[:, foot_idx].view(-1, 3)
        
        # 计算速度方向（在水平面上）
        # 箭头默认沿x轴正方向，我们需要根据实际速度方向旋转它
        # 使用atan2计算偏航角（yaw）
        yaw_angle = torch.atan2(foot_vel[:, 1], foot_vel[:, 0])
        
        # 创建四元数，绕z轴旋转yaw_angle角度
        zeros = torch.zeros_like(yaw_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, yaw_angle)
        
        # 缩放箭头长度（基于速度大小）
        # 基础长度为1.0，根据速度大小缩放
        default_scale = self.arrow_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(foot_vel.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(foot_vel, dim=1) * 3.0
        
        # 可视化箭头
        self.arrow_visualizer.visualize(
            translations=foot_pos,
            orientations=arrow_quat,
            scales=arrow_scale
        )

########################### 周期相位奖励 ###################################

class ICurve:
    """生成一个步态指示器I的曲线"""
    def __init__(self, duty_cycle, init_phase, freq, k = 10.0):
        """ 初始化步态指示器I的曲线

        Args:
            duty_cycle (_type_): 高电平占空比，一个周期内的百分比
            init_phase (_type_): 初相位，一个周期内的百分比
            freq (_type_): 频率
            k (float, optional): 曲线陡峭程度. Defaults to 10.0.
        """
        self.duty_cycle = torch.as_tensor(duty_cycle, dtype=torch.float32)
        self.init_phase = torch.as_tensor(init_phase, dtype=torch.float32, device=self.duty_cycle.device)
        self.freq = torch.as_tensor(freq, dtype=torch.float32, device=self.duty_cycle.device)
        self.k = torch.ones_like(self.duty_cycle, device=self.duty_cycle.device) * k
        self.b = torch.where( # 当stance rate=1.0时，直接让b等于k的两倍，保持直线
            self.duty_cycle == 1.0,
            self.k * 2.0,
            self.k * torch.sin(torch.pi * (self.duty_cycle - 0.5))
        )
        self.a = torch.pi / 2.0

    def __call__(self, t):
        """
        计算步态指示信号 f(x)=sigmoid(k·sin(2πf·(x+a+φ))+b)
        参数:
            t: torch.Tensor, 输入的时间序列，形状为 (batch_size, seq_len) 或 (seq_len,)
        返回:
            torch.Tensor, 生成的方波信号,形状与输入t相同
        """
        # 确保输入在正确的设备上
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self.duty_cycle.device)

        y = torch.sigmoid(self.k * torch.sin(2 * torch.pi * (self.freq * t - self.init_phase) + self.a) + self.b)
        return y

class BipedalICurve:
    """双足机器人的相位指示器"""
    def __init__(self, r_stance: torch.Tensor, offset:torch.Tensor, frequency: torch.Tensor, k: float = 10.0):
        """生成两个单足相位指示曲线

        Args:
            r_stance (_type_): 支撑相占比(高电平)，百分数
            t_offset (_type_): 两个足的步态偏移量，百分数
            cycle (float, optional): 步态周期，秒
            k (float, optional): 曲线陡峭程度. Defaults to 10.0.
        """

        # 分别创建双足的相位指示曲线
        zero_phase = torch.zeros_like(offset, device=offset.device)

        self.curve_left = ICurve(duty_cycle=r_stance, init_phase=zero_phase,freq=frequency,k=k)
        self.curve_right = ICurve(duty_cycle=r_stance, init_phase=offset,freq=frequency,k=k)

    def __call__(self, t: torch.Tensor):
        """ 分别返回双足相位指示曲线 """
        return self.curve_left(t), self.curve_right(t)
    
class QuadrupedICurve:
    """四足机器人的相位指示器"""
    def __init__(
            self, 
            r_stance: torch.Tensor, 
            rf_offset: torch.Tensor, 
            lb_offset: torch.Tensor, 
            rb_offset: torch.Tensor, 
            frequency: torch.Tensor,
            k: float = 10.0
        ):
        """生成四个相位指示曲线

        Args:
            r_stance (_type_): 支撑相占比
            rf_offset (_type_): rf相对lf的相位偏移量,百分比
            lb_offset (_type_): lb相对lf的相位偏移量,百分比
            rb_offset (_type_): rb相对lf的相位偏移量,百分比
            cycle (float, optional): 步态周期，秒
            k (float, optional): 曲线陡峭程度. Defaults to 10.0.
        """

        zero_phase = torch.zeros_like(rf_offset, device=rf_offset.device)

        # 分别创建双足的相位指示曲线
        self.curve_lf = ICurve(duty_cycle=r_stance, init_phase=zero_phase, freq=frequency, k=k)
        self.curve_rf = ICurve(duty_cycle=r_stance, init_phase=rf_offset, freq=frequency, k=k)
        self.curve_lb = ICurve(duty_cycle=r_stance, init_phase=lb_offset, freq=frequency, k=k)
        self.curve_rb = ICurve(duty_cycle=r_stance, init_phase=rb_offset, freq=frequency, k=k)

    def __call__(self, t: torch.Tensor):
        """ 分别返回四足相位指示曲线 """
        return self.curve_lf(t), self.curve_rf(t), self.curve_lb(t), self.curve_rb(t)

class BipedalGaitReward(ManagerTermBase):
    _env: ManagerBasedRLEnv
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg,env)

        # 获取命令值
        gait_command = self._env.command_manager.get_command("gait_command")
        # 记录当前与上一次命令值的缓冲区
        self.gait_command = torch.zeros_like(gait_command, device=self._env.device)
        self.last_gait_command = torch.zeros_like(gait_command, device=self._env.device)

        self.left_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.right_I = torch.zeros(self._env.num_envs, device=self._env.device)

    def __call__(
        self, 
        env: ManagerBasedRLEnv,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        left_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        right_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        k: float = 10.0,
        foot_height_tar: float = 0.1,
    ) -> torch.Tensor:
        # 获取当前的命令值
        self.gait_command = self._env.command_manager.get_command("gait_command")

        # 当有环境进行了命令重采样，重新初始化I曲线
        if not torch.equal(self.gait_command[:,:3],self.last_gait_command[:,:3]):
            self.bipedal_icurve = BipedalICurve(self.gait_command[:,0],self.gait_command[:,1],self.gait_command[:,2],k)
        
        # 获取当前时间
        now_time = self._env.episode_length_buf * self._env.step_dt

        # 获取当前双足的相位，1是支撑相，0是摆动相
        self.left_I, self.right_I = self.bipedal_icurve(now_time)

        left_rew = self.left_I * q_spd(env, left_asset_cfg) + (1-self.left_I) * (q_frc(env, left_sensor_cfg) + foot_height(env, foot_height_tar, left_asset_cfg)) / 2.0
        right_rew = self.right_I * q_spd(env, right_asset_cfg) + (1-self.right_I) * (q_frc(env, right_sensor_cfg) + foot_height(env, foot_height_tar, right_asset_cfg)) / 2.0

        # 记录上一次的命令值
        self.last_gait_command = self.gait_command.clone()

        return (left_rew + right_rew) / 2.0
    
class BipedalGaitEnsureReward(ManagerTermBase):
    _env: ManagerBasedRLEnv
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg,env)

        # 获取命令值
        gait_command = self._env.command_manager.get_command("gait_command")
        # 记录当前与上一次命令值的缓冲区
        self.gait_command = torch.zeros_like(gait_command, device=self._env.device)
        self.last_gait_command = torch.zeros_like(gait_command, device=self._env.device)

        self.left_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.right_I = torch.zeros(self._env.num_envs, device=self._env.device)

    def __call__(
        self, 
        env: ManagerBasedRLEnv,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        k: float = 10.0,
    ) -> torch.Tensor:
        # 获取当前的命令值
        self.gait_command = self._env.command_manager.get_command("gait_command")

        # 当有环境进行了命令重采样，重新初始化I曲线
        if not torch.equal(self.gait_command[:,:3],self.last_gait_command[:,:3]):
            self.bipedal_icurve = BipedalICurve(self.gait_command[:,0],self.gait_command[:,1],self.gait_command[:,2],k)
        
        # 获取当前时间
        now_time = self._env.episode_length_buf * self._env.step_dt

        # 获取当前双足的相位，1是支撑相，0是摆动相
        self.left_I, self.right_I = self.bipedal_icurve(now_time)

        # 计算当前双足的触地情况
        left_contact_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
        right_contact_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
        left_foot_frc_norm = left_contact_sensor.data.net_forces_w[:, left_sensor_cfg.body_ids, 2].norm(dim=-1)
        right_foot_frc_norm = right_contact_sensor.data.net_forces_w[:, right_sensor_cfg.body_ids, 2].norm(dim=-1)
        left_contact_flag = left_foot_frc_norm > 5.0
        right_contact_flag = right_foot_frc_norm > 5.0

        left_rew = 1 + 1.3 * (2 * self.left_I * left_contact_flag - self.left_I - left_contact_flag.int())
        right_rew = 1 + 1.3 * (2 * self.right_I * right_contact_flag - self.right_I - right_contact_flag.int())

        # 记录上一次的命令值
        self.last_gait_command = self.gait_command.clone()

        return (left_rew + right_rew) / 2.0

class QuadrupedGaitEnsureReward(ManagerTermBase):
    """四足机器人的相位确定奖励函数"""
    _env: ManagerBasedRLEnv
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 获取命令值
        gait_command = self._env.command_manager.get_command("gait_command")
        # 记录当前与上一次命令值的缓冲区
        self.gait_command = torch.zeros_like(gait_command, device=self._env.device)
        self.last_gait_command = torch.zeros_like(gait_command, device=self._env.device)

        self.LF_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.RF_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.LB_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.RB_I = torch.zeros(self._env.num_envs, device=self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lf_sensor_cfg: SceneEntityCfg,
        rf_sensor_cfg: SceneEntityCfg,
        lb_sensor_cfg: SceneEntityCfg,
        rb_sensor_cfg: SceneEntityCfg,
        k: float = 10.0,
    ) -> torch.Tensor:
        # 获取当前的命令值
        self.gait_command = self._env.command_manager.get_command("gait_command")

        # 当有环境进行了命令重采样，重新初始化I曲线
        # gait_command layout: [stance_rate, rf_offset, lb_offset, rb_offset, gait_frequency, ...]
        if not torch.equal(self.gait_command[:, :5], self.last_gait_command[:, :5]):
            self.quadruped_icurve = QuadrupedICurve(
                self.gait_command[:, 0],
                self.gait_command[:, 1],
                self.gait_command[:, 2],
                self.gait_command[:, 3],
                self.gait_command[:, 4],
                k
            )

        # 获取当前时间
        now_time = self._env.episode_length_buf * self._env.step_dt

        # 获取当前四足的相位，1是支撑相，0是摆动相
        self.LF_I, self.RF_I, self.LB_I, self.RB_I = self.quadruped_icurve(now_time)

        # 计算当前四足的触地情况
        lf_contact_sensor: ContactSensor = env.scene.sensors[lf_sensor_cfg.name]
        rf_contact_sensor: ContactSensor = env.scene.sensors[rf_sensor_cfg.name]
        lb_contact_sensor: ContactSensor = env.scene.sensors[lb_sensor_cfg.name]
        rb_contact_sensor: ContactSensor = env.scene.sensors[rb_sensor_cfg.name]
        
        lf_foot_frc_norm = lf_contact_sensor.data.net_forces_w[:, lf_sensor_cfg.body_ids, 2].norm(dim=-1)
        rf_foot_frc_norm = rf_contact_sensor.data.net_forces_w[:, rf_sensor_cfg.body_ids, 2].norm(dim=-1)
        lb_foot_frc_norm = lb_contact_sensor.data.net_forces_w[:, lb_sensor_cfg.body_ids, 2].norm(dim=-1)
        rb_foot_frc_norm = rb_contact_sensor.data.net_forces_w[:, rb_sensor_cfg.body_ids, 2].norm(dim=-1)
        
        lf_contact_flag = lf_foot_frc_norm > 5.0
        rf_contact_flag = rf_foot_frc_norm > 5.0
        lb_contact_flag = lb_foot_frc_norm > 5.0
        rb_contact_flag = rb_foot_frc_norm > 5.0

        # 计算每个足端的奖励
        lf_rew = 1 + 1.3 * (2 * self.LF_I * lf_contact_flag - self.LF_I - lf_contact_flag.int())
        rf_rew = 1 + 1.3 * (2 * self.RF_I * rf_contact_flag - self.RF_I - rf_contact_flag.int())
        lb_rew = 1 + 1.3 * (2 * self.LB_I * lb_contact_flag - self.LB_I - lb_contact_flag.int())
        rb_rew = 1 + 1.3 * (2 * self.RB_I * rb_contact_flag - self.RB_I - rb_contact_flag.int())

        # 记录上一次的命令值
        self.last_gait_command = self.gait_command.clone()

        return (lf_rew + rf_rew + lb_rew + rb_rew) / 4.0
    
class BipedalGaitRewardWithoutFeetHeight(BipedalGaitReward):
    _env: ManagerBasedRLEnv
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg,env)

    def __call__(
        self, 
        env: ManagerBasedRLEnv,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        left_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        right_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        k: float = 10.0,
    ) -> torch.Tensor:
        # 获取当前的命令值
        self.gait_command = self._env.command_manager.get_command("gait_command")

        # 当有环境进行了命令重采样，重新初始化I曲线
        if not torch.equal(self.gait_command[:,:3],self.last_gait_command[:,:3]):
            self.bipedal_icurve = BipedalICurve(self.gait_command[:,0],self.gait_command[:,1],self.gait_command[:,2],k)
        
        # 获取当前时间
        now_time = self._env.episode_length_buf * self._env.step_dt

        # 获取当前双足的相位，1是支撑相，0是摆动相
        self.left_I, self.right_I = self.bipedal_icurve(now_time)

        left_rew = self.left_I * q_spd(env, left_asset_cfg) + (1-self.left_I) * q_frc(env, left_sensor_cfg)
        right_rew = self.right_I * q_spd(env, right_asset_cfg) + (1-self.right_I) * q_frc(env, right_sensor_cfg)

        # 记录上一次的命令值
        self.last_gait_command = self.gait_command.clone()

        return (left_rew + right_rew) / 2.0

class QuadrupedGaitReward(ManagerTermBase):
    """四足机器人的步态奖励函数

    Args:
        GaitPhase (_type_): _description_
    """
    _env: ManagerBasedRLEnv
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg,env)
        # 获取命令值
        gait_command = self._env.command_manager.get_command("gait_command")
        # 记录当前与上一次命令值的缓冲区
        self.gait_command = torch.zeros_like(gait_command, device=self._env.device)
        self.last_gait_command = torch.zeros_like(gait_command, device=self._env.device)

        self.LF_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.RF_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.LB_I = torch.zeros(self._env.num_envs, device=self._env.device)
        self.RB_I = torch.zeros(self._env.num_envs, device=self._env.device)
    
    def __call__(
        self, 
        env: ManagerBasedRLEnv,
        lf_sensor_cfg: SceneEntityCfg,
        rf_sensor_cfg: SceneEntityCfg,
        lb_sensor_cfg: SceneEntityCfg,
        rb_sensor_cfg: SceneEntityCfg,
        
        lf_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        rf_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        lb_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        rb_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        k: float = 10.0,
        foot_height_tar: float = 0.05,
    ) -> torch.Tensor:
        """计算四足机器人的步态奖励"""
        # 获取当前的命令值
        self.gait_command = self._env.command_manager.get_command("gait_command")

        # 当有环境进行了命令重采样，重新初始化I曲线
        # gait_command layout: [stance_rate, rf_offset, lb_offset, rb_offset, gait_frequency, ...]
        if not torch.equal(self.gait_command[:, :5], self.last_gait_command[:, :5]):
            self.quadruped_icurve = QuadrupedICurve(
                self.gait_command[:,0],
                self.gait_command[:,1],
                self.gait_command[:,2],
                self.gait_command[:,3],
                self.gait_command[:,4],
                k,
            )

        # 获取当前时间
        now_time = self._env.episode_length_buf * self._env.step_dt

        # 获取当前四足的相位，1是支撑相，0是摆动相
        self.LF_I, self.RF_I, self.LB_I, self.RB_I = self.quadruped_icurve(now_time)

        # 获取四个足端是否触地
        lf_contact_sensor: ContactSensor = env.scene.sensors[lf_sensor_cfg.name]
        rf_contact_sensor: ContactSensor = env.scene.sensors[rf_sensor_cfg.name]
        lb_contact_sensor: ContactSensor = env.scene.sensors[lb_sensor_cfg.name]
        rb_contact_sensor: ContactSensor = env.scene.sensors[rb_sensor_cfg.name]
        lf_contact_flag = lf_contact_sensor.data.net_forces_w[:, lf_sensor_cfg.body_ids, :].norm(dim=-1).squeeze(-1) > 0.5
        rf_contact_flag = rf_contact_sensor.data.net_forces_w[:, rf_sensor_cfg.body_ids, :].norm(dim=-1).squeeze(-1) > 0.5
        lb_contact_flag = lb_contact_sensor.data.net_forces_w[:, lb_sensor_cfg.body_ids, :].norm(dim=-1).squeeze(-1) > 0.5
        rb_contact_flag = rb_contact_sensor.data.net_forces_w[:, rb_sensor_cfg.body_ids, :].norm(dim=-1).squeeze(-1) > 0.5

        # 计算每个足端的奖励
        lf_rew = self.LF_I * lf_contact_flag * q_spd(env, lf_asset_cfg) + (1-self.LF_I) * (q_frc(env, lf_sensor_cfg) + foot_height(env, foot_height_tar, lf_asset_cfg)) / 2.0
        rf_rew = self.RF_I * rf_contact_flag * q_spd(env, rf_asset_cfg) + (1-self.RF_I) * (q_frc(env, rf_sensor_cfg) + foot_height(env, foot_height_tar, rf_asset_cfg)) / 2.0
        lb_rew = self.LB_I * lb_contact_flag * q_spd(env, lb_asset_cfg) + (1-self.LB_I) * (q_frc(env, lb_sensor_cfg) + foot_height(env, foot_height_tar, lb_asset_cfg)) / 2.0
        rb_rew = self.RB_I * rb_contact_flag * q_spd(env, rb_asset_cfg) + (1-self.RB_I) * (q_frc(env, rb_sensor_cfg) + foot_height(env, foot_height_tar, rb_asset_cfg)) / 2.0
        # 记录上一次的命令值
        self.last_gait_command = self.gait_command.clone()
        
        return (lf_rew + rf_rew + lb_rew + rb_rew) / 4.0

class QuadrupedGaitReward(ManagerTermBase):
    """
    Lab中自带的周期相位奖励函数，但是思路是让对角线腿的悬空，触地时间接近
    Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward
    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

########################### 基于周期相位的对称奖励 ###################################

class HumanSymmetricReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 获取命令值
        gait_command = self._env.command_manager.get_command("gait_command")
        # 记录当前与上一次命令值的缓冲区
        self.gait_command = torch.zeros_like(gait_command, device=self._env.device)
        self.last_gait_command = torch.zeros_like(gait_command, device=self._env.device)  

    def _param_init(self, r_stance, t_offset, gait_cycle: float, joint_num):
        # 初始化相位
        # super()._param_init(gait_cycle)
        self.r_stance = r_stance
        self.t_offset = t_offset

        # 计算一个步态周期有多少个step, 目前是40 steps
        # step_dt:0.02s, physics_dt:0.005s
        self.cycle_step_num = int(1.0 / self.gait_command[0,2] / self._env.step_dt)

        # 创建左右腿关节位置缓冲区 shape=[2*40, 4096, 6]
        # TODO: cycle time 重采样后会有问题
        self.left_joint_pos_buf = torch.zeros(2*self.cycle_step_num, self._env.num_envs, joint_num, device=self._env.device)
        self.right_joint_pos_buf = torch.zeros(2*self.cycle_step_num, self._env.num_envs, joint_num, device=self._env.device)

    def __call__(
            self, 
            env: ManagerBasedRLEnv, 
            std: float,
            left_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            right_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """计算人形机器人运动对称奖励

        Args:
            env (ManagerBasedRLEnv): 环境类，不用管
            std (float): 高斯核的标准差
            left_asset_cfg (SceneEntityCfg, optional): 左腿关节列表
            right_asset_cfg (SceneEntityCfg, optional): 右腿关节列表

        Returns:
            torch.Tensor: 奖励值 shape=[env]
        """
        # 获取当前的命令值
        self.gait_command = self._env.command_manager.get_command("gait_command")
        # 当有环境进行了命令重采样，重新进行变量赋值
        if not torch.equal(self.gait_command[:,:3],self.last_gait_command[:,:3]):
            joint_num = len(left_asset_cfg.joint_ids)
            self._param_init(self.gait_command[:,0],self.gait_command[:,1],self.gait_command[:,2],joint_num)
        
        # 记录上一次的命令值
        self.last_gait_command = self.gait_command.clone()
        # 读取当前机器人的关节角度
        asset: RigidObject = env.scene[left_asset_cfg.name]
        left_joint_pos = asset.data.joint_pos[:, left_asset_cfg.joint_ids].clone() # 获取关节位置
        asset: RigidObject = env.scene[right_asset_cfg.name]
        right_joint_pos = asset.data.joint_pos[:, right_asset_cfg.joint_ids].clone() # 获取关节位置

        # 对缓冲区进行移位，0号是最新的，-1是最旧的
        self.left_joint_pos_buf = torch.roll(self.left_joint_pos_buf, shifts=1, dims=0)
        self.right_joint_pos_buf = torch.roll(self.right_joint_pos_buf, shifts=1, dims=0)

        # 将0号赋值为最新的关节角
        self.left_joint_pos_buf[0] = left_joint_pos
        self.right_joint_pos_buf[0] = right_joint_pos

        # 位移右腿缓冲区到左腿的对应相位
        shift_num = int((self.cycle_step_num*self.t_offset[0]).item())
        right_shift_buf = torch.roll(self.right_joint_pos_buf,shifts=-shift_num,dims=0)
        # 取第一个周期的buf进行比较
        left_compare_buf = self.left_joint_pos_buf[0:self.cycle_step_num,:]
        right_compare_buf = right_shift_buf[0:self.cycle_step_num,:]
        # 计算左右腿的关节位置差
        error = torch.square(left_compare_buf - right_compare_buf) # shape=[history,env,joints]
        error = torch.mean(error, dim=0) # shape=[env,joints]
        error = torch.mean(error, dim=-1) # shape=[env]
        # 计算奖励值
        return torch.exp(-error / std**2) 
    
class QuadrupedSymmetricReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 获取命令值
        gait_command = self._env.command_manager.get_command("gait_command")
        # 记录当前与上一次命令值的缓冲区
        self.gait_command = torch.zeros_like(gait_command, device=self._env.device)
        self.last_gait_command = torch.zeros_like(gait_command, device=self._env.device)  

    def _param_init(self, r_stance, rf_offset, lb_offset, rb_offset, gait_cycle: float, joint_num):
        # 初始化相位
        # super()._param_init(gait_cycle)
        self.r_stance = r_stance
        self.rf_offset = rf_offset
        self.lb_offset = lb_offset
        self.rb_offset = rb_offset

        # 计算一个步态周期有多少个step, 目前是40 steps
        # step_dt:0.02s, physics_dt:0.005s
        self.cycle_step_num = int(self.gait_command[0,4] / self._env.step_dt)

        # 创建左右腿关节位置缓冲区 shape=[2*40, 4096, 6]
        # TODO: cycle time 重采样后会有问题
        self.left_joint_pos_buf = torch.zeros(2*self.cycle_step_num, self._env.num_envs, joint_num, device=self._env.device)
        self.right_joint_pos_buf = torch.zeros(2*self.cycle_step_num, self._env.num_envs, joint_num, device=self._env.device)

    def __call__(
            self, 
            env: ManagerBasedRLEnv, 
            std: float,
            left_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            right_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """计算四足机器人运动对称奖励

        Args:
            env (ManagerBasedRLEnv): 环境类，不用管
            std (float): 高斯核的标准差
            left_asset_cfg (SceneEntityCfg, optional): 左腿关节列表
            right_asset_cfg (SceneEntityCfg, optional): 右腿关节列表

        Returns:
            torch.Tensor: 奖励值 shape=[env]
        """
        # 获取当前的命令值
        self.gait_command = self._env.command_manager.get_command("gait_command")
        # 当有环境进行了命令重采样，重新进行变量赋值
        if not torch.equal(self.gait_command[:,:5],self.last_gait_command[:,:5]):
            joint_num = len(left_asset_cfg.joint_ids)
            self._param_init(self.gait_command[:,0],self.gait_command[:,1],self.gait_command[:,2],self.gait_command[:,3],self.gait_command[:,4],joint_num)
        
        # 记录上一次的命令值
        self.last_gait_command = self.gait_command.clone()
        # 读取当前机器人的关节角度
        asset: RigidObject = env.scene[left_asset_cfg.name]
        left_joint_pos = asset.data.joint_pos[:, left_asset_cfg.joint_ids].clone() # 获取关节位置
        asset: RigidObject = env.scene[right_asset_cfg.name]
        right_joint_pos = asset.data.joint_pos[:, right_asset_cfg.joint_ids].clone() # 获取关节位置

        # 对缓冲区进行移位，0号是最新的，-1是最旧的
        self.left_joint_pos_buf = torch.roll(self.left_joint_pos_buf, shifts=1, dims=0)
        self.right_joint_pos_buf = torch.roll(self.right_joint_pos_buf, shifts=1, dims=0)

        # 将0号赋值为最新的关节角
        self.left_joint_pos_buf[0] = left_joint_pos
        self.right_joint_pos_buf[0] = right_joint_pos

        # 位移右腿缓冲区到左腿的对应相位
        shift_num = int((self.cycle_step_num*self.rf_offset[0]).item())
        right_shift_buf = torch.roll(self.right_joint_pos_buf,shifts=-shift_num,dims=0)
        # 取第一个周期的buf进行比较
        left_compare_buf = self.left_joint_pos_buf[0:self.cycle_step_num,:]
        right_compare_buf = right_shift_buf[0:self.cycle_step_num,:]
        # 计算左右腿的关节位置差
        error = torch.square(left_compare_buf - right_compare_buf) # shape=[history,env,joints]
        error = torch.mean(error, dim=0) # shape=[env,joints]
        error = torch.mean(error, dim=-1) # shape=[env]
        # 计算奖励值
        return torch.exp(-error / std**2) 

########################### 奖励单腿摆动 ###################################

def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float, 
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground
    
    Args:
        env: 环境实例
        asset_cfg: 资产配置（通常是脚部链接）
        target_height: 目标离地高度
        std: 高斯核的标准差
        tanh_mult: tanh函数的倍数
        sensor_cfg: 可选的RayCaster传感器配置，用于获取地形高度。如果提供，目标高度会加上地形高度
    
    Returns:
        奖励值张量
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取脚部的世界坐标z值
    foot_z_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # shape: [num_envs, num_feet]
    
    # 如果指定了传感器，获取地形高度
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        # 获取射线击中点的z坐标（地形高度）
        terrain_height = sensor.data.ray_hits_w[:, :, 2]  # shape: [num_envs, num_rays]
        # 对每个环境取平均地形高度
        terrain_height_mean = torch.mean(terrain_height, dim=1, keepdim=True)  # shape: [num_envs, 1]
        # 目标高度需要加上地形高度
        effective_target_height = terrain_height_mean + target_height
    else:
        # 没有传感器时，使用固定的目标高度
        effective_target_height = target_height
    
    # 计算脚部高度与目标高度的误差
    foot_z_target_error = torch.square(foot_z_pos - effective_target_height)
    
    # 计算脚部水平速度的tanh值（用于识别摆动相）
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    
    # 组合误差和速度
    reward = foot_z_target_error * foot_velocity_tanh
    
    return torch.exp(-torch.sum(reward, dim=1) / std)

## 未分类
def human_shoulder_trajectory_l2( 
        env: ManagerBasedRLEnv,
        left_shoulder_cfg: SceneEntityCfg,
        right_shoulder_cfg: SceneEntityCfg,
        left_elbow_cfg: SceneEntityCfg,
        right_elbow_cfg: SceneEntityCfg,
        shoulder_pitch_center: float,
        elbow_center: float,
        std: float = 0.5,
    ):
        # 获取命令值
        gait_command = env.command_manager.get_command("gait_command")
        vel_command = env.command_manager.get_command("base_velocity")
        # 获取当前时间
        now_time = env.episode_length_buf * env.step_dt
        
        # 根据速度指令缩放摆手的幅度
        vel_cmd_range_x = env.command_manager.cfg.base_velocity.ranges.lin_vel_x
        vel_cmd_range_y = env.command_manager.cfg.base_velocity.ranges.lin_vel_y
        vel_cmd_range_z = env.command_manager.cfg.base_velocity.ranges.ang_vel_z

        vel_cmd_amplitude_x = (vel_command[:,0] - vel_cmd_range_x[0]) / (vel_cmd_range_x[1] - vel_cmd_range_x[0]+1e-6)  # 避免除以0
        vel_cmd_amplitude_y = (vel_command[:,1] - vel_cmd_range_y[0]) / (vel_cmd_range_y[1] - vel_cmd_range_y[0]+1e-6)
        vel_cmd_amplitude_z = (vel_command[:,2] - vel_cmd_range_z[0]) / (vel_cmd_range_z[1] - vel_cmd_range_z[0]+1e-6)

        vel_cmd_amplitude_mean = (vel_cmd_amplitude_x + vel_cmd_amplitude_y + vel_cmd_amplitude_z) / 3.0

        shoulder_amplitude = 0.305 * vel_cmd_amplitude_mean
        elbow_amplitude = 0.2125 * vel_cmd_amplitude_mean

        # 计算左右手的轨迹 
        frequency = 1 / gait_command[:,2]
        phase_offset = gait_command[:,1] * gait_command[:,2]
        dir_offset = torch.where( # 避免后退时同手同脚
            vel_command[:,0] > 0,
            0,
            0.5*gait_command[:,2]
        )
        left_shoulder_trajectory = shoulder_amplitude * torch.cos(2 * torch.pi * frequency * (now_time + dir_offset)) + shoulder_pitch_center
        right_shoulder_trajectory = shoulder_amplitude * torch.cos(2 * torch.pi * frequency * (now_time + dir_offset + phase_offset)) + shoulder_pitch_center
        left_elbow_trajectory = elbow_amplitude * torch.cos(2 * torch.pi * frequency * (now_time + dir_offset)) + elbow_center
        right_elbow_trajectory = elbow_amplitude * torch.cos(2 * torch.pi * frequency * (now_time + dir_offset + phase_offset)) + elbow_center
        # 获取左右手当前的关节位置
        left_shoulder_asset: Articulation = env.scene[left_shoulder_cfg.name]
        right_shoulder_asset: Articulation = env.scene[right_shoulder_cfg.name]
        left_shoulder_pos = left_shoulder_asset.data.joint_pos[:, left_shoulder_cfg.joint_ids].squeeze(-1)
        right_shoulder_pos = right_shoulder_asset.data.joint_pos[:, right_shoulder_cfg.joint_ids].squeeze(-1)
        left_elbow_asset: Articulation = env.scene[left_elbow_cfg.name]
        right_elbow_asset: Articulation = env.scene[right_elbow_cfg.name]
        left_elbow_pos = left_elbow_asset.data.joint_pos[:, left_elbow_cfg.joint_ids].squeeze(-1)
        right_elbow_pos = right_elbow_asset.data.joint_pos[:, right_elbow_cfg.joint_ids].squeeze(-1)
        
        # 如果速度小于阈值，则希望手臂保持零位
        
        cmd_threshold_flag = torch.norm(vel_command,dim=1) < 0.1
        left_shoulder_trajectory = torch.where(
            cmd_threshold_flag,
            left_shoulder_asset.data.default_joint_pos[:, left_shoulder_cfg.joint_ids].squeeze(-1),
            left_shoulder_trajectory
        )
        right_shoulder_trajectory = torch.where(
            cmd_threshold_flag,
            right_shoulder_asset.data.default_joint_pos[:, right_shoulder_cfg.joint_ids].squeeze(-1),
            right_shoulder_trajectory
        )
        left_elbow_trajectory = torch.where(
            cmd_threshold_flag,
            left_elbow_asset.data.default_joint_pos[:, left_elbow_cfg.joint_ids].squeeze(-1),
            left_elbow_trajectory
        )
        right_elbow_trajectory = torch.where(
            cmd_threshold_flag,
            right_elbow_asset.data.default_joint_pos[:, right_elbow_cfg.joint_ids].squeeze(-1),
            right_elbow_trajectory
        )

        # 计算左右手臂的误差
        left_arm_err = torch.square(left_shoulder_pos - left_shoulder_trajectory) + torch.square(left_elbow_pos - left_elbow_trajectory)
        right_arm_err = torch.square(right_shoulder_pos - right_shoulder_trajectory) + torch.square(right_elbow_pos - right_elbow_trajectory)

        # return (torch.exp(-left_shoulder_err / std**2) + torch.exp(-right_shoulder_err / std**2)) / 2
        return (-left_arm_err / std**2 + -right_arm_err / std**2) / 2.0 + 1.0

def biped_distance_y_l2(
        env: ManagerBasedRLEnv, 
        min_distance: float, 
        max_distance: float, 
        min_distance_standing: float = None,
        max_distance_standing: float = None,
        command_name: str = "base_velocity",
        velocity_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.tensor:
    """
    对双足机器人,如果两个部件的y方向距离过大或者过小就会被惩罚
    支持在行走和站立状态下使用不同的距离范围
    
    Args:
        env: 环境实例
        min_distance: 行走时的最小距离
        max_distance: 行走时的最大距离
        min_distance_standing: 站立时的最小距离（如果为None，则使用min_distance）
        max_distance_standing: 站立时的最大距离（如果为None，则使用max_distance）
        command_name: 速度指令名称
        velocity_threshold: 判断站立/行走的速度阈值
        asset_cfg: 资产配置
        
    权重需要是负数
    """
    # 如果没有指定站立状态的距离范围，使用行走时的范围
    if min_distance_standing is None:
        min_distance_standing = min_distance
    if max_distance_standing is None:
        max_distance_standing = max_distance
    
    asset: RigidObject = env.scene[asset_cfg.name] # asset就是robot
    link_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 1] # 获取y坐标
    link_y_distance = torch.abs(link_pos[:, 0] - link_pos[:, 1]) # y坐标的差值
    
    # 获取速度指令，判断是站立还是行走
    vel_command = env.command_manager.get_command(command_name)
    
    # 计算速度指令的模（xy平面 + 角速度）
    total_vel_norm = torch.norm(vel_command[:, :2],dim=1)
    
    # 判断是否为站立状态（速度指令接近0）
    is_standing = total_vel_norm < velocity_threshold  # shape: [num_envs]
    
    # 根据状态选择对应的距离范围
    min_dist = torch.where(is_standing, 
                          torch.tensor(min_distance_standing, device=env.device),
                          torch.tensor(min_distance, device=env.device))
    max_dist = torch.where(is_standing,
                          torch.tensor(max_distance_standing, device=env.device),
                          torch.tensor(max_distance, device=env.device))
    
    # 计算超出范围的部分
    d_min = torch.clamp(link_y_distance - min_dist, -1.0, 0)
    d_max = torch.clamp(link_y_distance - max_dist, 0, 1.0)
    
    return (torch.abs(d_min) + torch.abs(d_max)) / 2

########################### MSE直接模仿数据集 ###################################

def mimic(
    env: ManagerBasedRLEnv,
    command_name: str = "mimic_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    """模仿奖励, 来自slw"""

    command = env.command_manager.get_term(command_name)
    pos_command = command.command

    asset: RigidObject = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # 只看y关节
    # joint_pos = joint_pos[:,[2, 3, 4, 8, 9, 10]]
    # pos_command = pos_command[:,[2, 3, 4, 8, 9, 10]]

    mimic_error = torch.mean(torch.abs(pos_command - joint_pos), dim=1)

    reward = torch.exp(-mimic_error / std**2)

    # return torch.clip(reward, -2, 1)
    return reward

#################################################################################################
####################################### 下面是正则化函数 #########################################
#################################################################################################

########################### 机身高度跟踪衍生奖励函数 ###################################

def lin_vel_z_exp(
        env: ManagerBasedRLEnv, 
        std: float = 0.3,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """
    使用高斯核与l2范数一起给出z方向线速度的惩罚
    权重需为正
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    rew = torch.exp(-torch.square(asset.data.root_lin_vel_b[:, 2]) / std**2)
    return  rew

def ang_vel_xy_exp(
        env: ManagerBasedRLEnv, 
        std: float = 0.45,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """
    使用高斯核与l2范数一起给出xy方向角速度的惩罚
    权重需为正
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_xy = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    rew = torch.exp(-ang_vel_xy / std**2) 
    return rew

def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    惩罚action的二阶微分
    权重应为负数
    """

    return torch.sum(torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager._prev_prev_action), dim=1)

def dof_smooth_l2(
        env: ManagerBasedRLEnv, 
        c_dof_v: float,
        c_dof_a: float,
        c_tor: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """
    使用l2范数对关节角度进行平滑化
    权重应为负数

    Args:
        env (ManagerBasedRLEnv): 环境类
        c_dof (float): 关节速度项的权重系数,正值
        c_ddof (float): 关节加速度项的权重系数,正值
        c_tor (float): 力矩项的权重系数,正值
        asset_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("robot").
    """
    asset: Articulation = env.scene[asset_cfg.name]

    dof_v = asset.data.joint_vel[:, asset_cfg.joint_ids]
    dof_a = asset.data.joint_acc[:, asset_cfg.joint_ids]
    dof_tor = asset.data.applied_torque[:, asset_cfg.joint_ids]

    return torch.sum(c_dof_v * torch.square(dof_v) + c_dof_a * torch.square(dof_a) + c_tor * torch.square(dof_tor), dim=1)

def body_smooth_l2(
        env: ManagerBasedRLEnv, 
        c_body_a: float,
        c_feet_a: float,
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        feet_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """
    使用l2范数对身体位置和姿态进行平滑化
    权重应为负数

    Args:
        env (ManagerBasedRLEnv): 环境类
        c_body_a (float): base加速度项权重系数,正值
        c_feet_a (float): 足端加速度项项的权重系数,正值
        base_asset_cfg (SceneEntityCfg, optional): base link配置项
        feet_asset_cfg (SceneEntityCfg, optional): feet link配置项
    """
    base_asset: RigidObject = env.scene[base_asset_cfg.name]
    feet_asset: RigidObject = env.scene[feet_asset_cfg.name]

    body_lin_acc_b = math_utils.quat_apply_inverse(base_asset.data.root_link_quat_w, base_asset.data.body_lin_acc_w[:, base_asset_cfg.body_ids, :].squeeze(1))
    feet_lin_acc_b = torch.zeros([env.num_envs,len(feet_asset_cfg.body_ids),3],dtype=torch.float32,device=env.device)
    # 对每一只脚进行坐标系转换
    for i in range(len(feet_asset_cfg.body_ids)):
        feet_lin_acc_b[:,i,:] = math_utils.quat_apply_inverse(feet_asset.data.root_link_quat_w, feet_asset.data.body_lin_acc_w[:, feet_asset_cfg.body_ids[i], :].squeeze(1))
    

    return torch.sum(c_body_a * torch.square(body_lin_acc_b) + c_feet_a * torch.sum(torch.square(feet_lin_acc_b),dim=1), dim=1)

def joint_pos_soft_limits(env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """关节位置软限位"""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = (
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids])
        - asset.data.joint_pos_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # 将差距限制在[0-1]避免巨大的惩罚
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)

def joint_vel_soft_limits(env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """关节速度软限位"""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # 将差距限制在[0-1]避免巨大的惩罚
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)

def joint_tor_soft_limits(env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """关节力矩软限位"""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = (
        torch.abs(asset.data.computed_torque[:, asset_cfg.joint_ids]) # computed_torque 是被限幅前的力矩
        - asset.data.joint_effort_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # 将差距限制在[0-1]避免巨大的惩罚
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)

def stand_still_without_cmd(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return (
        torch.sum(torch.abs(diff_angle), dim=1) * command
    )

def stand_still_without_cmd_vel(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """在速度命令接近0的时候惩罚关节速度"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取关节转速
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return (torch.sum(torch.abs(joint_vel), dim=1) * command)

def stand_still_without_cmd_vel_l1_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """在速度命令接近0的时候惩罚关节速度（使用L1+L2混合惩罚）"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取关节转速
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # 判断命令是否接近0（显式转换为float）
    command = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1).float()
    # 计算L1+L2混合惩罚（归一化）
    penalty = torch.mean(torch.abs(joint_vel) + torch.square(joint_vel), dim=1)
    return penalty * command

def stand_still_without_cmd_base_vel(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """在速度命令接近0的时候惩罚base线速度"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return (torch.sum(torch.abs(asset.data.root_lin_vel_b), dim=1) * command)

def feet_stumble(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    """
    惩罚足端x、y方向的受力,避免撞击地形
    权重应该是负数
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.any(
        torch.norm(contacts_force[:, :, :2], dim=-1) > \
        3 * contacts_force[:, :, 2],
        dim=1
    )

def flat_orientation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1)

def body_orientation_l2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """要求特定的link保持水平，Lab自带的函数只针对base_link，对髋、胸分离的机器人效果不够好

    Args:
        env (ManagerBasedRLEnv): _description_
        asset_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("robot").

    Returns:
        torch.Tensor: _description_
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize mechanical energy usage."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_vel) * torch.abs(applied_torque), dim=-1)

def body_lin_vel_z_exp(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        lambda_exp: float = 2.0,
    ) -> torch.Tensor:
    """Reward keeping the selected body link vertical velocity near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_z = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids[0], 2]
    return torch.exp(-lambda_exp * torch.square(vel_z))

def body_ang_vel_xy_exp(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        lambda_exp: float = 1.0,
    ) -> torch.Tensor:
    """Reward keeping the selected body link roll/pitch angular velocity near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :2]
    return torch.exp(-lambda_exp * torch.sum(torch.square(ang_vel_xy), dim=1))

def body_upright_bonus_exp(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        lambda_exp: float = 4.0,
    ) -> torch.Tensor:
    """Reward the selected body link for staying upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    projected_gravity = math_utils.quat_apply_inverse(body_quat, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.exp(-lambda_exp * torch.square(projected_gravity[:, :2])), dim=1)

def body_height_exp(
        env: ManagerBasedRLEnv,
        target_height: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        lambda_exp: float = 10.0,
    ) -> torch.Tensor:
    """Reward the selected body link for staying near a target height."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    height_error = current_height - target_height
    return torch.exp(-lambda_exp * torch.abs(height_error))

def action_rate_l2_clipped(env: ManagerBasedRLEnv, max_penalty: float = 100.0) -> torch.Tensor:
    """Penalize action changes with an upper bound."""
    action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.clamp(action_rate, max=max_penalty)

def joint_target_l1(
        env: ManagerBasedRLEnv,
        targets: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize selected joints for deviating from explicit target angles."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(asset.num_joints))[joint_ids]

    target_pos = asset.data.default_joint_pos[:, joint_ids].clone()
    for col, joint_id in enumerate(joint_ids):
        joint_name = asset.joint_names[joint_id]
        if joint_name in targets:
            target_pos[:, col] = targets[joint_name]

    return torch.sum(torch.abs(asset.data.joint_pos[:, joint_ids] - target_pos), dim=1)

def object_upright_bonus_exp(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        lambda_exp: float = 4.0,
    ) -> torch.Tensor:
    """Reward a rigid object for staying upright."""
    obj: RigidObject = env.scene[object_cfg.name]
    projected_gravity = obj.data.projected_gravity_b
    return torch.sum(torch.exp(-lambda_exp * torch.square(projected_gravity[:, :2])), dim=1)

def object_lin_vel_z_exp(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        lambda_exp: float = 2.0,
    ) -> torch.Tensor:
    """Reward a rigid object for keeping vertical linear velocity near zero."""
    obj: RigidObject = env.scene[object_cfg.name]
    vel_z = obj.data.body_lin_vel_w[:, 0, 2]
    return torch.exp(-lambda_exp * torch.square(vel_z))

def object_ang_vel_xy_exp(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        lambda_exp: float = 1.0,
    ) -> torch.Tensor:
    """Reward a rigid object for keeping roll/pitch angular velocity near zero."""
    obj: RigidObject = env.scene[object_cfg.name]
    ang_vel_xy = obj.data.body_ang_vel_w[:, 0, :2]
    return torch.exp(-lambda_exp * torch.sum(torch.square(ang_vel_xy), dim=1))

def object_ang_vel_exp(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        lambda_exp: float = 1.0,
    ) -> torch.Tensor:
    """Reward a rigid object for keeping full angular velocity near zero."""
    obj: RigidObject = env.scene[object_cfg.name]
    ang_vel = obj.data.body_ang_vel_w[:, 0, :]
    return torch.exp(-lambda_exp * torch.sum(torch.square(ang_vel), dim=1))

def object_rel_rot_exp(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        reference_asset_cfg: SceneEntityCfg,
        target_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        std: float = 0.5,
    ) -> torch.Tensor:
    """Reward a rigid object for staying near a target orientation in a reference body frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    reference: Articulation | RigidObject = env.scene[reference_asset_cfg.name]

    if isinstance(reference, Articulation):
        ref_quat = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]
    else:
        ref_quat = reference.data.root_quat_w

    rel_quat = math_utils.quat_mul(math_utils.quat_inv(ref_quat), obj.data.root_quat_w)
    target = torch.tensor(target_quat, dtype=rel_quat.dtype, device=rel_quat.device).unsqueeze(0).expand_as(rel_quat)
    rot_error = quat_error_magnitude(rel_quat, target)
    return torch.exp(-torch.square(rot_error) / std**2)

def object_rel_pos_exp(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        reference_asset_cfg: SceneEntityCfg,
        target_pos: tuple[float, float, float],
        std: float = 0.2,
    ) -> torch.Tensor:
    """Reward a rigid object for staying near a target pose relative to a robot link."""
    obj: RigidObject = env.scene[object_cfg.name]
    reference: Articulation | RigidObject = env.scene[reference_asset_cfg.name]

    if isinstance(reference, Articulation):
        ref_pos = reference.data.body_link_pos_w[:, reference_asset_cfg.body_ids[0]]
        ref_quat = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]
    else:
        ref_pos = reference.data.root_pos_w
        ref_quat = reference.data.root_quat_w

    rel_pos = math_utils.quat_apply_inverse(ref_quat, obj.data.root_pos_w - ref_pos)
    target = torch.tensor(target_pos, dtype=rel_pos.dtype, device=rel_pos.device).unsqueeze(0)
    pos_error = torch.sum(torch.square(rel_pos - target), dim=1)
    return torch.exp(-pos_error / std**2)

def body_object_rel_pos_exp(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        target_offsets: tuple[tuple[float, float, float], ...],
        std: float = 0.2,
    ) -> torch.Tensor:
    """Reward selected robot bodies for staying near target offsets in an object's frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    num_bodies = body_pos_w.shape[1]
    rel_pos_w = body_pos_w - obj.data.root_pos_w.unsqueeze(1)
    obj_quat_w = obj.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    rel_pos = math_utils.quat_apply_inverse(
        obj_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    target = torch.tensor(target_offsets, dtype=rel_pos.dtype, device=rel_pos.device).unsqueeze(0)
    pos_error = torch.sum(torch.square(rel_pos - target), dim=(1, 2))
    return torch.exp(-pos_error / std**2)

def body_body_rel_pos_exp(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        reference_asset_cfg: SceneEntityCfg,
        target_positions: tuple[tuple[float, float, float], ...],
        std: float = 0.2,
    ) -> torch.Tensor:
    """Reward selected robot bodies for staying near target positions in a reference body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    reference: Articulation = env.scene[reference_asset_cfg.name]

    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    ref_pos_w = reference.data.body_link_pos_w[:, reference_asset_cfg.body_ids[0]]
    ref_quat_w = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]

    num_bodies = body_pos_w.shape[1]
    rel_pos_w = body_pos_w - ref_pos_w.unsqueeze(1)
    ref_quat_w = ref_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    rel_pos = math_utils.quat_apply_inverse(
        ref_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    target = torch.tensor(target_positions, dtype=rel_pos.dtype, device=rel_pos.device).unsqueeze(0)
    pos_error = torch.sum(torch.square(rel_pos - target), dim=(1, 2))
    return torch.exp(-pos_error / std**2)

def body_inside_box_penalty(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        reference_asset_cfg: SceneEntityCfg,
        box_center: tuple[float, float, float],
        box_half_size: tuple[float, float, float],
        margin: float = 0.0,
    ) -> torch.Tensor:
    """Penalty depth for selected body points inside an axis-aligned box in a reference body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    reference: Articulation = env.scene[reference_asset_cfg.name]

    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    ref_pos_w = reference.data.body_link_pos_w[:, reference_asset_cfg.body_ids[0]]
    ref_quat_w = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]

    num_bodies = body_pos_w.shape[1]
    rel_pos_w = body_pos_w - ref_pos_w.unsqueeze(1)
    ref_quat_w = ref_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    rel_pos = math_utils.quat_apply_inverse(
        ref_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    center = torch.tensor(box_center, dtype=rel_pos.dtype, device=rel_pos.device)
    half_size = torch.tensor(box_half_size, dtype=rel_pos.dtype, device=rel_pos.device) + margin
    depth = half_size - torch.abs(rel_pos - center)
    inside = torch.all(depth > 0.0, dim=-1)
    penetration_depth = torch.where(
        inside,
        torch.min(depth, dim=-1).values,
        torch.zeros_like(depth[..., 0]),
    )
    return torch.sum(penetration_depth, dim=1)

def desired_contacts_count(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """Reward sustained filtered contacts across the contact sensor history."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :].norm(dim=-1) > threshold
    return contacts.sum(dim=(1, 2, 3)).float()

def contact_force_exp(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        lambda_exp: float = 0.01,
    ) -> torch.Tensor:
    """Reward gentle filtered contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :]
    total_force = torch.norm(contact_forces, dim=-1).sum(dim=(1, 2))
    return torch.exp(-lambda_exp * total_force)

#################################################################################################
################################## 给其他奖励函数调用的辅助函数 ###################################
#################################################################################################

def foot_height(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """足端高度越接近特定值,奖励越高"""
    # 获取机器人
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取足端高度
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].squeeze(-1)
    # 如果使用了地形，需要减去地形高度
    if "height_scanner" in env.scene.sensors:
        terrain_height = torch.mean(env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2],dim=1)
        terrain_height = torch.clip(terrain_height,-10.0,10.0)
        foot_height -= terrain_height
    # 返回奖励值
    return -torch.abs(foot_height - target) / 0.1 + 1.0

def q_frc(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """足端力越接近0越奖励"""
    # 获取力矩传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取足端力范数 
    foot_frc_norm = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1).squeeze(-1)
    # 返回奖励值
    # return torch.exp(-torch.square(foot_frc_norm) / 100)
    return torch.clip(-torch.square(foot_frc_norm) / 100.0 + 1.0, min=-1.0, max=1.0)  # 确保奖励值在[-1,1]之间

def q_spd(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """足端速度越接近0越奖励"""
    # 获取机器人
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取足端速度
    foot_spd_norm = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].norm(dim=-1).squeeze(-1)
    # 返回奖励值
    return torch.exp(-2 * torch.square(foot_spd_norm))
