from __future__ import annotations

import math
import torch
import json
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from .commands import HAND_KINEMATIC_DIM, HAND_REFERENCE_DIM, MotionCommand
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def two_hand_reference_tracking_error(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Directional two-hand pose and velocity errors from a replay command."""
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "tracking_error"):
        raise TypeError(f"Command term '{command_name}' does not expose tracking_error().")
    return command_term.tracking_error()


def two_hand_reference_kinematics(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return only p/q/v/w from the two-hand CSV command.

    The command is laid out per hand, so it must be reshaped before removing
    force and moment.  Slicing the flattened command directly would mix the
    left-hand wrench with the right-hand kinematic state.
    """
    command = env.command_manager.get_command(command_name)
    command = command.reshape(env.num_envs, 2, HAND_REFERENCE_DIM)
    return command[:, :, :HAND_KINEMATIC_DIM].reshape(env.num_envs, -1)


def _s2_virtual_spring(env: ManagerBasedRLEnv):
    spring = getattr(env, "_virtual_spring", None)
    if spring is not None:
        return spring
    if getattr(env, "_virtual_spring_initializing", False):
        # ObservationManager calls every term once while the parent
        # environment constructor is still discovering observation shapes.
        return None
    raise RuntimeError("S2 virtual-force observation requested without a virtual spring manager.")


def two_hand_virtual_force_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Two target environment-on-hand forces in the current torso frame."""
    spring = _s2_virtual_spring(env)
    if spring is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    return spring.target_force_observation()


def two_hand_actual_virtual_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged two-hand virtual spring force in the current torso frame.

    This is the known force applied by the virtual environment, not a
    wrist-force sensor or physical contact-force measurement.
    """
    spring = _s2_virtual_spring(env)
    if spring is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    return spring.actual_force_observation()


def two_hand_virtual_spring_deflection(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Two spring equilibrium-minus-palm displacements in the torso frame."""
    spring = _s2_virtual_spring(env)
    if spring is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    return spring.spring_deflection_observation()


def two_hand_force_control_axes(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Two force-control axis-direction vectors in the current torso frame."""
    spring = _s2_virtual_spring(env)
    if spring is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    return spring.force_control_axis_observation()


def _ensure_s2_action_component_buffers(env: ManagerBasedRLEnv) -> None:
    """Lazily create policy-action buffers used by S2 observations/rewards."""
    action_dim = env.action_manager.total_action_dim
    specs = (
        "_s2_previous_base_action",
        "_s2_last_residual_action",
        "_s2_previous_residual_action",
    )
    for name in specs:
        value = getattr(env, name, None)
        if value is None or value.shape != (env.num_envs, action_dim):
            setattr(
                env,
                name,
                torch.zeros(env.num_envs, action_dim, device=env.device, dtype=torch.float32),
            )


def previous_base_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the frozen S1 actor's previous action, not the previous total action."""
    _ensure_s2_action_component_buffers(env)
    return env._s2_previous_base_action


def previous_residual_action(
    env: ManagerBasedRLEnv,
    action_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the residual that was physically applied on the preceding step."""
    _ensure_s2_action_component_buffers(env)
    residual = env._s2_last_residual_action
    if action_indices is not None:
        residual = residual[:, list(action_indices)]
    return residual


class obs_history(ManagerTermBase):
    """返回历史观测值，包括当前帧观测值"""

    def __init__(self, cfg: ObsTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.obs_his_buf = None

    def __call__(self, env: ManagerBasedEnv, history_length: int, obs_len: int, obs_name: str) -> torch.Tensor:

        # 第一次执行需要初始化缓存变量
        if self.obs_his_buf is None:
            # 初始化历史观测值缓存
            self.obs_len = obs_len
            self.obs_his_buf = torch.zeros(env.num_envs, history_length * obs_len, device=env.device)
        

        if hasattr(env, "obs_buf") and env.obs_buf:
            new_obs = env.obs_buf[obs_name]
            self.obs_his_buf = torch.roll(self.obs_his_buf, self.obs_len, dims=1)
            self.obs_his_buf[:, 0:self.obs_len] = new_obs
        return self.obs_his_buf

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.obs_his_buf[env_ids] = 0

def body_pos_z(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """The height of specified bodies above the terrain surface.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their heights returned.
        For flat terrain, height is relative to env origin. For rough terrain,
        sensor readings adjust the height to account for the terrain.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.
        sensor_cfg: The RayCaster sensor for terrain height. If None, uses env origin.

    Returns:
        The heights of bodies above terrain [num_env, num_bodies].
        Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Get body z positions in world frame
    body_z = asset.data.body_pose_w[:, asset_cfg.body_ids, 2]  # [num_envs, num_bodies]
    
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        # Get terrain height from sensor
        terrain_height = torch.clip(torch.mean(sensor.data.ray_hits_w[..., 2], dim=1), -10.0, 10.0)  # [num_envs]
        # Calculate height above terrain for each body
        height_above_terrain = body_z - terrain_height.unsqueeze(1)
    else:
        # Fallback: relative to environment origin
        height_above_terrain = body_z - env.scene.env_origins[:, 2].unsqueeze(1)
    
    return height_above_terrain.reshape(env.num_envs, -1)

def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def body_ang_vel_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The angular velocity of body in the base frame.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their velocities returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The angular velocity of bodies in articulation [num_env, 3 * num_bodies].
        Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body angular velocities in world frame
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :].squeeze()
    
    # project to base frame
    root_quat_b_w = asset.data.root_quat_w
    
    # Apply inverse rotation
    body_ang_vel_b = math_utils.quat_apply_inverse(root_quat_b_w, body_ang_vel_w)
    
    return body_ang_vel_b.view(env.num_envs, -1)

def rigid_object_projected_gravity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Projected gravity of a rigid object in its body frame."""
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.projected_gravity_b

def asset_rel_pos(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    reference_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target asset position expressed in the reference asset frame."""
    target = env.scene[target_asset_cfg.name]
    reference = env.scene[reference_asset_cfg.name]

    if isinstance(target, Articulation):
        target_pos_w = target.data.body_link_pos_w[:, target_asset_cfg.body_ids[0]]
    else:
        target_pos_w = target.data.root_pos_w

    if isinstance(reference, Articulation):
        ref_pos_w = reference.data.body_link_pos_w[:, reference_asset_cfg.body_ids[0]]
        ref_quat_w = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]
    else:
        ref_pos_w = reference.data.root_pos_w
        ref_quat_w = reference.data.root_quat_w

    return math_utils.quat_apply_inverse(ref_quat_w, target_pos_w - ref_pos_w)


def asset_rel_quat(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    reference_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target asset orientation relative to the reference asset frame."""
    target = env.scene[target_asset_cfg.name]
    reference = env.scene[reference_asset_cfg.name]

    if isinstance(target, Articulation):
        target_quat_w = target.data.body_link_quat_w[:, target_asset_cfg.body_ids[0]]
    else:
        target_quat_w = target.data.root_quat_w

    if isinstance(reference, Articulation):
        ref_quat_w = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]
    else:
        ref_quat_w = reference.data.root_quat_w

    rel_quat = math_utils.quat_mul(math_utils.quat_inv(ref_quat_w), target_quat_w)
    return math_utils.quat_unique(rel_quat)

def asset_rel_lin_vel(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    reference_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target asset linear velocity relative to reference asset, expressed in the reference frame."""
    target = env.scene[target_asset_cfg.name]
    reference = env.scene[reference_asset_cfg.name]

    if isinstance(target, Articulation):
        target_lin_vel_w = target.data.body_link_lin_vel_w[:, target_asset_cfg.body_ids[0]]
    else:
        target_lin_vel_w = target.data.body_lin_vel_w[:, 0]

    if isinstance(reference, Articulation):
        ref_lin_vel_w = reference.data.body_link_lin_vel_w[:, reference_asset_cfg.body_ids[0]]
        ref_quat_w = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]
    else:
        ref_lin_vel_w = reference.data.body_lin_vel_w[:, 0]
        ref_quat_w = reference.data.root_quat_w

    return math_utils.quat_apply_inverse(ref_quat_w, target_lin_vel_w - ref_lin_vel_w)

def asset_rel_ang_vel(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    reference_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target asset angular velocity relative to reference asset, expressed in the reference frame."""
    target = env.scene[target_asset_cfg.name]
    reference = env.scene[reference_asset_cfg.name]

    if isinstance(target, Articulation):
        target_ang_vel_w = target.data.body_link_ang_vel_w[:, target_asset_cfg.body_ids[0]]
    else:
        target_ang_vel_w = target.data.body_ang_vel_w[:, 0]

    if isinstance(reference, Articulation):
        ref_ang_vel_w = reference.data.body_link_ang_vel_w[:, reference_asset_cfg.body_ids[0]]
        ref_quat_w = reference.data.body_link_quat_w[:, reference_asset_cfg.body_ids[0]]
    else:
        ref_ang_vel_w = reference.data.body_ang_vel_w[:, 0]
        ref_quat_w = reference.data.root_quat_w

    return math_utils.quat_apply_inverse(ref_quat_w, target_ang_vel_w - ref_ang_vel_w)

def body_rel_pos_to_object(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Selected articulation body positions expressed in the rigid object's frame."""
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
    return rel_pos.reshape(env.num_envs, -1)

def filtered_contact_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flatten filtered contact force vectors from a contact sensor."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, :, :]
    return force_matrix.reshape(env.num_envs, -1)

########################### amp用 ###################################
class AmpExpertDataSampler(ManagerTermBase):
    """
    从AMP专家数据集中按权重随机采样当前帧和下一帧
    将当前帧和下一帧横向拼接在一起
    """

    def __init__(self, cfg: ObsTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # 数据加载相关
        self.trajectories = []
        self.trajectory_weights = []
        self.trajectory_lens = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        
        # 性能优化：缓存轨迹帧数（NumPy数组，用于向量化采样）
        self._traj_num_frames_np = None
        
        # 将在__call__中初始化
        self._initialized = False

    def _load_motion_data(self, motion_files_or_dir: str | list[str]):
        """
        加载motion文件
        
        Args:
            motion_files_or_dir: 可以是：
                - 文件夹路径字符串：自动加载该文件夹下所有.txt文件
                - 文件路径列表：加载指定的文件列表
        """
        # 1. 确定要加载的文件列表
        if isinstance(motion_files_or_dir, str):
            # 如果是字符串，判断是文件夹还是文件
            motion_path = Path(motion_files_or_dir)
            
            # 如果是相对路径，从当前文件所在目录解析
            if not motion_path.is_absolute():
                # 获取当前配置文件所在目录
                import inspect
                frame = inspect.currentframe()
                if frame and frame.f_back and frame.f_back.f_back:
                    caller_frame = frame.f_back.f_back
                    if 'self' in caller_frame.f_locals:
                        env = caller_frame.f_locals.get('env')
                        if env and hasattr(env, 'cfg') and hasattr(env.cfg, '__class__'):
                            cfg_module = inspect.getfile(env.cfg.__class__)
                            cfg_dir = Path(cfg_module).parent
                            motion_path = (cfg_dir / motion_files_or_dir).resolve()
                
                # 如果上面的方法失败，使用环境配置文件的路径
                if not motion_path.exists():
                    motion_path = Path(__file__).parent.parent / motion_files_or_dir
                    motion_path = motion_path.resolve()
            
            if motion_path.is_dir():
                # 文件夹：加载所有.txt文件
                motion_files = sorted(motion_path.glob("*.txt"))
                motion_files = [str(f) for f in motion_files]
                print(f"从文件夹加载: {motion_path}")
                print(f"找到 {len(motion_files)} 个motion文件")
            elif motion_path.is_file():
                # 单个文件
                motion_files = [str(motion_path)]
            else:
                raise ValueError(f"路径不存在: {motion_path}")
        else:
            # 文件列表
            motion_files = motion_files_or_dir
        
        if not motion_files:
            raise ValueError("motion_files列表为空，请提供有效的motion文件路径或文件夹")
        
        # 2. 加载文件
        for motion_file in motion_files:
            if not Path(motion_file).exists():
                print(f"警告: 文件不存在 {motion_file}, 跳过")
                continue
                
            with open(motion_file, 'r') as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                
                # 转换为tensor并存储
                self.trajectories.append(
                    torch.tensor(motion_data, dtype=torch.float32, device=self._env.device)
                )
                
                # 存储元数据
                self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))
                
                print(f"已加载 {motion_file}: {motion_data.shape[0]}帧, 时长{traj_len:.2f}s, "
                      f"权重={motion_json.get('MotionWeight', 1.0)}")
        
        if len(self.trajectories) == 0:
            raise ValueError("没有成功加载任何motion文件")
        
        # 归一化权重
        total_weight = sum(self.trajectory_weights)
        self.trajectory_weights = [w / total_weight for w in self.trajectory_weights]
        
        # 转换为numpy数组方便采样
        self.trajectory_weights = np.array(self.trajectory_weights)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)
        
        # 性能优化：缓存轨迹帧数（用于向量化采样）
        self._traj_num_frames_np = np.array([traj.shape[0] for traj in self.trajectories], dtype=np.int64)
        
        print(f"总共加载 {len(self.trajectories)} 个motion文件")
        print(f"权重分布: {self.trajectory_weights}")

    def _sample_frames(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        按权重随机采样当前帧和下一帧（优化版本：向量化 + 预分配）
        
        性能优化：
        1. 向量化随机数生成（避免循环调用 np.random.randint）
        2. 预分配输出张量（避免动态列表扩展）
        3. 缓存轨迹元数据（避免重复访问 .shape）
        
        Returns:
            current_frames: shape (batch_size, frame_dim)
            next_frames: shape (batch_size, frame_dim)
        """
        # 1. 按权重采样轨迹索引（向量化操作）
        traj_idxs = np.random.choice(
            len(self.trajectories), 
            size=batch_size, 
            p=self.trajectory_weights,
            replace=True
        )
        
        # 2. 向量化生成随机帧索引
        # 获取每个采样轨迹的帧数
        sampled_traj_num_frames = self._traj_num_frames_np[traj_idxs]
        
        # 生成随机帧索引（确保 < num_frames - 1，以保证有下一帧）
        # 使用 np.maximum 避免除零和负数
        frame_idxs = np.floor(
            np.random.rand(batch_size) * np.maximum(1, sampled_traj_num_frames - 1)
        ).astype(np.int64)
        
        # 3. 预分配输出张量（避免动态扩展）
        frame_dim = self.trajectories[0].shape[1]
        current_frames = torch.empty(batch_size, frame_dim, device=self._env.device, dtype=torch.float32)
        next_frames = torch.empty(batch_size, frame_dim, device=self._env.device, dtype=torch.float32)
        
        # 4. 批量索引赋值（由于轨迹长度不同，这个循环无法避免）
        for i in range(batch_size):
            traj_idx = traj_idxs[i]
            frame_idx = frame_idxs[i]
            traj = self.trajectories[traj_idx]
            
            current_frames[i] = traj[frame_idx]
            # 安全地获取下一帧（避免越界）
            next_idx = min(frame_idx + 1, traj.shape[0] - 1)
            next_frames[i] = traj[next_idx]
        
        return current_frames, next_frames

    def __call__(
        self, 
        env: ManagerBasedEnv,
        motion_files: str | list[str],
        return_next_frame: bool = False,
    ) -> torch.Tensor:
        """
        采样expert数据
        
        Args:
            env: 环境实例
            motion_files: motion文件路径（文件夹路径或文件列表）
            return_next_frame: 是否返回下一帧（True则返回拼接的当前帧+下一帧）
            
        Returns:
            如果return_next_frame=False: shape (num_envs, frame_dim)
            如果return_next_frame=True: shape (num_envs, frame_dim * 2)
        """
        # 首次调用时加载数据
        if not self._initialized:
            self._load_motion_data(motion_files)
            self._initialized = True
        
        # 采样当前帧和下一帧
        current_frames, next_frames = self._sample_frames(env.num_envs)
        
        if return_next_frame:
            # 拼接当前帧和下一帧
            return torch.cat([current_frames, next_frames], dim=-1)
        else:
            # 只返回当前帧
            return current_frames

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """重置时不需要特殊处理，每次调用都会重新采样"""
        pass

def bodies_pos_order_r(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        bodies_order: str | list[str] = [],
) -> torch.Tensor:
    """
    Computes the bodies pos observation order in robot root frame from the environment's state.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(name_keys=bodies_order, preserve_order=True)
    bodies_pos_rel = robot.data.body_pos_w[:, body_ids, :3] - robot.data.root_pos_w.unsqueeze(1)
    num_envs, num_bodies, _ = bodies_pos_rel.shape
    bodies_pos_flat = bodies_pos_rel.reshape(num_envs * num_bodies, 3)

    root_quat = robot.data.root_state_w[:, 3:7]
    root_quat_expanded = root_quat.unsqueeze(1).repeat(1, num_bodies, 1).reshape(num_envs * num_bodies, 4)

    bodies_pos_root_flat = math_utils.quat_apply_inverse(root_quat_expanded, bodies_pos_flat)

    bodies_pos_root = bodies_pos_root_flat.reshape(num_envs, num_bodies, 3)

    return bodies_pos_root.reshape(num_envs, -1)  # Flatten for observation

########################### 高程图采样用 ###################################

class height_scan_sampled(ManagerTermBase):
    """间隔抽帧的高程图观测函数
    
    支持间隔抽帧数和最终总帧数设置，每隔一定帧数抽取一帧，
    返回指定长度的高程图时间序列。使用索引指针法实现高效环形缓冲区。
    
    Args:
        interval_frames: 间隔抽帧数（如10表示每10帧抽1帧）
        total_frames: 最终返回的总帧数（如5表示返回5帧序列）
        sensor_cfg: 传感器配置
    
    Returns:
        形状为 (num_envs, total_frames, height, width) 的张量，按时间顺序排列
    """
    
    def __init__(self, cfg: ObsTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # 配置参数
        self.interval_frames = cfg.params["interval_frames"]  # 间隔抽帧数
        self.total_frames = cfg.params["total_frames"]  # 最终总帧数
        
        # 传感器配置缓存
        self._sensor_cfg = cfg.params["sensor_cfg"]
        
        # 直接初始化传感器参数
        sensor: RayCaster = env.scene[self._sensor_cfg.name]
        scan_height_meter = sensor.cfg.pattern_cfg.size[0]
        scan_width_meter = sensor.cfg.pattern_cfg.size[1]
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        self._scan_height_points = math.ceil((float(scan_height_meter) + 1.0e-9) / float(scan_resolution))
        self._scan_width_points = math.ceil((float(scan_width_meter) + 1.0e-9) / float(scan_resolution))

        # 初始化缓冲区
        self.sampled_buffer = torch.zeros(
            env.num_envs, 
            self.total_frames, 
            self._scan_height_points, 
            self._scan_width_points,
            device=env.device, 
            dtype=torch.float32
        )
        
        # 初始化指针和计数器
        self.write_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)  # 写入位置指针，范围 [0, total_frames-1]
        self.frame_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)  # 当前帧计数器
        self.sample_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)  # 已采样帧数计数器（用于处理初始未满情况）
        
        # 标记是否已获取第一帧
        self._first_frame_captured = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def _get_ordered_buffer(self) -> torch.Tensor:
        """获取按时间顺序排列的缓冲区数据
        
        Returns:
            形状为 (num_envs, total_frames, height, width) 的张量
        """
        batch_size = self.sampled_buffer.shape[0]
        
        # 创建读取索引：从写指针位置开始，按时间顺序读取
        # 例如：write_index=3, total_frames=5，读取顺序为 [3,4,0,1,2]
        time_indices = torch.arange(self.total_frames, device=self.sampled_buffer.device)
        read_indices = (time_indices.unsqueeze(0) + self.write_index.unsqueeze(1)) % self.total_frames
        
        # 使用高级索引获取按时间顺序的数据
        batch_indices = torch.arange(batch_size, device=self.sampled_buffer.device).unsqueeze(1)
        ordered_buffer = self.sampled_buffer[batch_indices, read_indices]
        
        return ordered_buffer

    def __call__(
            self, 
            env: ManagerBasedEnv,
            sensor_cfg: SceneEntityCfg = None,
            interval_frames: int = None,
            total_frames: int = None,
            ) -> torch.Tensor:
        """执行间隔抽帧的高程图观测
        
        Args:
            env: 环境实例
            sensor_cfg: 传感器配置（可选，如果提供则覆盖初始化时的配置）
            interval_frames: 间隔抽帧数（可选，如果提供则覆盖初始化时的配置）
            total_frames: 最终返回的总帧数（可选，如果提供则覆盖初始化时的配置）
            
        Returns:
            形状为 (num_envs, total_frames, height, width) 的张量，按时间顺序排列
        """
        
        # 获取当前高程图数据
        sensor: RayCaster = env.scene[self._sensor_cfg.name]
        current_scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]
        
        # 重新整形为 (num_envs, height, width)
        current_scan = current_scan.view(env.num_envs, self._scan_height_points, self._scan_width_points)
        
        # 对于还未获取第一帧的环境，用当前帧填充所有时间槽
        first_time_envs = ~self._first_frame_captured
        if first_time_envs.any():
            first_time_ids = first_time_envs.nonzero(as_tuple=True)[0]
            # 用当前帧填充所有时间槽
            self.sampled_buffer[first_time_ids] = current_scan[first_time_ids].unsqueeze(1).repeat(1, self.total_frames, 1, 1)
            # 更新写指针和采样计数
            self.write_index[first_time_ids] = self.total_frames  # 写指针指向末尾，表示已填满
            self.sample_count[first_time_ids] = self.total_frames
            # 标记已获取第一帧
            self._first_frame_captured[first_time_ids] = True
        
        # 检查是否需要采样
        should_sample = (self.frame_counter % self.interval_frames == 0)
        
        if should_sample.any():
            # 只对已获取第一帧的环境进行采样
            should_sample = should_sample & self._first_frame_captured
            env_ids_to_sample = should_sample.nonzero(as_tuple=True)[0]
            
            if len(env_ids_to_sample) > 0:
                # 高效写入：直接写入到写指针位置，无需移动数据
                current_write_indices = self.write_index[env_ids_to_sample] % self.total_frames
                self.sampled_buffer[env_ids_to_sample, current_write_indices] = current_scan[env_ids_to_sample]
                
                # 更新写指针（循环）
                self.write_index[env_ids_to_sample] = (current_write_indices + 1) % self.total_frames

        # 更新帧计数器
        self.frame_counter += 1
        
        # 返回按时间顺序排列的采样结果
        return self._get_ordered_buffer()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """重置指定环境的缓冲区和计数器
        
        Args:
            env_ids: 要重置的环境ID列表，如果为None则重置所有环境
        """
        if env_ids is None:
            # 重置所有环境
            self.sampled_buffer.zero_()
            self.write_index.zero_()
            self.frame_counter.zero_()
            self.sample_count.zero_()
            self._first_frame_captured.zero_()
        else:
            # 重置指定环境
            if len(env_ids) > 0:
                # env_ids_tensor = torch.tensor(env_ids, device=self.sampled_buffer.device).detach().clone()
                env_ids_tensor = env_ids.detach().clone()
                self.sampled_buffer[env_ids_tensor] = 0
                self.write_index[env_ids_tensor] = 0
                self.frame_counter[env_ids_tensor] = 0
                self.sample_count[env_ids_tensor] = 0
                self._first_frame_captured[env_ids_tensor] = False

    def get_sample_count(self) -> torch.Tensor:
        """获取每个环境已采样的帧数
        
        Returns:
            形状为 (num_envs,) 的张量，表示每个环境已采样的帧数
        """
        return self.sample_count.clone()

class height_scan_sampled_delayed(ManagerTermBase):
    """间隔抽帧 + 延迟采样的高程图观测函数

    说明:
        1) 每个 step 都把最新高程图写入原始缓冲区。
        2) 间隔采样从第 1 帧开始：t % interval_frames == 0。
        3) 延迟采样从第 (1 + delay_frames) 帧开始：
           (t - delay_frames) % interval_frames == 0，且 t >= delay_frames。
        4) 输出维度与 height_scan_sampled 一致，保持 view/顺序不变。

    Args:
        interval_frames: 间隔抽帧数（如10表示每10帧抽1帧）
        total_frames: 最终返回的总帧数（如5表示返回5帧序列）
        delay_frames: 延迟采样帧数（如2表示从第3帧开始间隔采样）
        sensor_cfg: 传感器配置

    Returns:
        形状为 (num_envs, total_frames, height, width) 的张量，按时间顺序排列
    """

    def __init__(self, cfg: ObsTerm, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 配置参数
        self.interval_frames = cfg.params["interval_frames"]
        self.total_frames = cfg.params["total_frames"]
        self.delay_frames = cfg.params.get("delay_frames", 0)

        # 传感器配置缓存
        self._sensor_cfg = cfg.params["sensor_cfg"]

        # 直接初始化传感器参数
        sensor: RayCaster = env.scene[self._sensor_cfg.name]
        scan_height_meter = sensor.cfg.pattern_cfg.size[0]
        scan_width_meter = sensor.cfg.pattern_cfg.size[1]
        scan_resolution = sensor.cfg.pattern_cfg.resolution
        self._scan_height_points = math.ceil((float(scan_height_meter) + 1.0e-9) / float(scan_resolution))
        self._scan_width_points = math.ceil((float(scan_width_meter) + 1.0e-9) / float(scan_resolution))

        # 采样缓冲区（与原实现一致）
        self.sampled_buffer = torch.zeros(
            env.num_envs,
            self.total_frames,
            self._scan_height_points,
            self._scan_width_points,
            device=env.device,
            dtype=torch.float32,
        )

        # 原始缓冲区：每个 step 写入最新高程图（用于延迟采样）
        self._raw_buffer_len = max(1, self.delay_frames + 1)
        self.raw_buffer = torch.zeros(
            env.num_envs,
            self._raw_buffer_len,
            self._scan_height_points,
            self._scan_width_points,
            device=env.device,
            dtype=torch.float32,
        )

        # 初始化指针和计数器
        self.write_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.raw_write_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.frame_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.sample_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        # 标记是否已获取第一帧
        self._first_frame_captured = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def _get_ordered_buffer(self) -> torch.Tensor:
        """获取按时间顺序排列的缓冲区数据"""
        batch_size = self.sampled_buffer.shape[0]

        time_indices = torch.arange(self.total_frames, device=self.sampled_buffer.device)
        read_indices = (time_indices.unsqueeze(0) + self.write_index.unsqueeze(1)) % self.total_frames

        batch_indices = torch.arange(batch_size, device=self.sampled_buffer.device).unsqueeze(1)
        ordered_buffer = self.sampled_buffer[batch_indices, read_indices]
        return ordered_buffer

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = None,
        interval_frames: int = None,
        total_frames: int = None,
        delay_frames: int = None,
    ) -> torch.Tensor:
        """执行间隔抽帧 + 延迟采样的高程图观测"""

        # 获取当前高程图数据
        sensor: RayCaster = env.scene[self._sensor_cfg.name]
        current_scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]
        current_scan = current_scan.view(env.num_envs, self._scan_height_points, self._scan_width_points)

        # 首帧：用当前帧填充所有时间槽，保证输出稳定
        first_time_envs = ~self._first_frame_captured
        if first_time_envs.any():
            first_time_ids = first_time_envs.nonzero(as_tuple=True)[0]
            self.sampled_buffer[first_time_ids] = (
                current_scan[first_time_ids].unsqueeze(1).repeat(1, self.total_frames, 1, 1)
            )
            self.write_index[first_time_ids] = self.total_frames
            self.sample_count[first_time_ids] = self.total_frames
            self._first_frame_captured[first_time_ids] = True

            # 同时初始化原始缓冲区
            self.raw_buffer[first_time_ids] = (
                current_scan[first_time_ids].unsqueeze(1).repeat(1, self._raw_buffer_len, 1, 1)
            )
            self.raw_write_index[first_time_ids] = 0

        # 每个 step 都写入原始缓冲区
        raw_write_indices = self.raw_write_index % self._raw_buffer_len
        batch_indices = torch.arange(env.num_envs, device=env.device)
        self.raw_buffer[batch_indices, raw_write_indices] = current_scan
        self.raw_write_index = (raw_write_indices + 1) % self._raw_buffer_len

        # 计算是否采样（考虑延迟）
        if self.delay_frames > 0:
            valid_delay = self.frame_counter >= self.delay_frames
            should_sample = valid_delay & ((self.frame_counter - self.delay_frames) % self.interval_frames == 0)
        else:
            should_sample = (self.frame_counter % self.interval_frames == 0)

        if should_sample.any():
            should_sample = should_sample & self._first_frame_captured
            env_ids_to_sample = should_sample.nonzero(as_tuple=True)[0]

            if len(env_ids_to_sample) > 0:
                # 获取延迟帧（基于原始缓冲区）
                delayed_indices = (raw_write_indices - self.delay_frames) % self._raw_buffer_len
                delayed_scans = self.raw_buffer[batch_indices, delayed_indices]

                current_write_indices = self.write_index[env_ids_to_sample] % self.total_frames
                self.sampled_buffer[env_ids_to_sample, current_write_indices] = delayed_scans[env_ids_to_sample]
                self.write_index[env_ids_to_sample] = (current_write_indices + 1) % self.total_frames

        self.frame_counter += 1
        return self._get_ordered_buffer()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """重置指定环境的缓冲区和计数器"""
        if env_ids is None:
            self.sampled_buffer.zero_()
            self.raw_buffer.zero_()
            self.write_index.zero_()
            self.raw_write_index.zero_()
            self.frame_counter.zero_()
            self.sample_count.zero_()
            self._first_frame_captured.zero_()
        else:
            if len(env_ids) > 0:
                env_ids_tensor = env_ids.detach().clone()
                self.sampled_buffer[env_ids_tensor] = 0
                self.raw_buffer[env_ids_tensor] = 0
                self.write_index[env_ids_tensor] = 0
                self.raw_write_index[env_ids_tensor] = 0
                self.frame_counter[env_ids_tensor] = 0
                self.sample_count[env_ids_tensor] = 0
                self._first_frame_captured[env_ids_tensor] = False

    def get_sample_count(self) -> torch.Tensor:
        """获取每个环境已采样的帧数"""
        return self.sample_count.clone()
