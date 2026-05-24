from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply


class BoxLinearVelocityCommand(CommandTerm):
    """Sample commanded box linear velocity in world frame: [vx, vy, vz]."""

    cfg: "BoxLinearVelocityCommandCfg"

    def __init__(self, cfg: "BoxLinearVelocityCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_w

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.lin_vel_z)

    def _update_command(self):
        pass


@configclass
class BoxLinearVelocityCommandCfg(CommandTermCfg):
    """Linear velocity command config for the carried box."""

    class_type: type = BoxLinearVelocityCommand

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        lin_vel_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


def box_edge_projection_exp(
    env: ManagerBasedRLEnv,
    box_name: str = "box",
    edge_axis: str = "x",
    edge_length: float = 0.4,
    target_length: float = 0.4,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward large when a box edge's ground-plane projection is near target_length."""
    axis_w = _box_local_axis_w(env, box_name, edge_axis)
    projected_length = torch.linalg.norm(axis_w[:, :2], dim=1) * edge_length
    return torch.exp(-torch.square(projected_length - target_length) / (std**2))


def box_length_vector_exp(
    env: ManagerBasedRLEnv,
    box_name: str = "box",
    edge_axis: str = "x",
    edge_length: float = 0.4,
    target_length: float = 0.4,
    std: float = 0.05,
) -> torch.Tensor:
    """Backward-compatible name for the box edge projection reward."""
    return box_edge_projection_exp(env, box_name, edge_axis, edge_length, target_length, std)


def box_height_l2(env: ManagerBasedRLEnv, box_name: str = "box", target_height: float = 0.95) -> torch.Tensor:
    """Squared box height error."""
    box: RigidObject = env.scene[box_name]
    return torch.square(box.data.root_pos_w[:, 2] - target_height)


def box_up_axis_l2(env: ManagerBasedRLEnv, box_name: str = "box", up_axis: str = "z") -> torch.Tensor:
    """Penalty for the box local up axis tilting away from world +Z."""
    up_w = _box_local_axis_w(env, box_name, up_axis)
    return torch.sum(torch.square(up_w[:, :2]), dim=1)


def track_box_lin_vel_xyz_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    box_name: str = "box",
    std: float = 0.5,
) -> torch.Tensor:
    """Reward box linear velocity tracking for commanded [vx, vy, vz]."""
    box: RigidObject = env.scene[box_name]
    vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name) - box.data.root_lin_vel_w), dim=1)
    return torch.exp(-vel_error / (std**2))


def _get_step_dt(env: ManagerBasedRLEnv) -> float:
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    return float(env.cfg.sim.dt * env.cfg.decimation)


def _get_env_buffer(env: ManagerBasedRLEnv, name: str, shape: torch.Size) -> torch.Tensor:
    if (not hasattr(env, name)) or getattr(env, name).shape != shape:
        setattr(env, name, torch.zeros(shape, device=env.device, dtype=torch.float32))
    return getattr(env, name)


def _box_local_axis_w(env: ManagerBasedRLEnv, box_name: str, axis: str) -> torch.Tensor:
    box: RigidObject = env.scene[box_name]
    if axis == "x":
        axis_local = (1.0, 0.0, 0.0)
    elif axis == "y":
        axis_local = (0.0, 1.0, 0.0)
    elif axis == "z":
        axis_local = (0.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported box local axis: {axis}")
    axis_tensor = torch.tensor(axis_local, device=env.device, dtype=torch.float32).repeat(
        box.data.root_quat_w.shape[0], 1
    )
    return quat_apply(box.data.root_quat_w, axis_tensor)


def box_linear_acc_l2(env: ManagerBasedRLEnv, box_name: str = "box") -> torch.Tensor:
    """Linear-acceleration penalty for the carried rigid object."""
    box: RigidObject = env.scene[box_name]
    vel = box.data.root_lin_vel_w
    dt = _get_step_dt(env)

    prev_vel = _get_env_buffer(env, "_coop_prev_box_lin_vel_w", vel.shape)
    acc = (vel - prev_vel) / dt

    just_reset = env.episode_length_buf == 0
    if torch.any(just_reset):
        acc[just_reset] = 0.0

    prev_vel.copy_(vel)
    return torch.sum(torch.square(acc), dim=1)


def box_angular_acc_l2(env: ManagerBasedRLEnv, box_name: str = "box") -> torch.Tensor:
    """Angular-acceleration penalty for the carried rigid object."""
    box: RigidObject = env.scene[box_name]
    ang_vel = box.data.root_ang_vel_w
    dt = _get_step_dt(env)

    prev_ang_vel = _get_env_buffer(env, "_coop_prev_box_ang_vel_w", ang_vel.shape)
    ang_acc = (ang_vel - prev_ang_vel) / dt

    just_reset = env.episode_length_buf == 0
    if torch.any(just_reset):
        ang_acc[just_reset] = 0.0

    prev_ang_vel.copy_(ang_vel)
    return torch.sum(torch.square(ang_acc), dim=1)


def box_linear_jerk_l2(env: ManagerBasedRLEnv, box_name: str = "box") -> torch.Tensor:
    """Linear-jerk penalty for the carried rigid object."""
    box: RigidObject = env.scene[box_name]
    vel = box.data.root_lin_vel_w
    dt = _get_step_dt(env)

    prev_vel = _get_env_buffer(env, "_coop_prev_box_lin_vel_w_for_jerk", vel.shape)
    prev_acc = _get_env_buffer(env, "_coop_prev_box_lin_acc_w", vel.shape)

    acc = (vel - prev_vel) / dt
    jerk = (acc - prev_acc) / dt

    just_reset = env.episode_length_buf == 0
    if torch.any(just_reset):
        acc[just_reset] = 0.0
        jerk[just_reset] = 0.0

    prev_vel.copy_(vel)
    prev_acc.copy_(acc)
    return torch.sum(torch.square(jerk), dim=1)


def box_angular_jerk_l2(env: ManagerBasedRLEnv, box_name: str = "box") -> torch.Tensor:
    """Angular-jerk penalty for the carried rigid object."""
    box: RigidObject = env.scene[box_name]
    ang_vel = box.data.root_ang_vel_w
    dt = _get_step_dt(env)

    prev_ang_vel = _get_env_buffer(env, "_coop_prev_box_ang_vel_w_for_jerk", ang_vel.shape)
    prev_ang_acc = _get_env_buffer(env, "_coop_prev_box_ang_acc_w", ang_vel.shape)

    ang_acc = (ang_vel - prev_ang_vel) / dt
    ang_jerk = (ang_acc - prev_ang_acc) / dt

    just_reset = env.episode_length_buf == 0
    if torch.any(just_reset):
        ang_acc[just_reset] = 0.0
        ang_jerk[just_reset] = 0.0

    prev_ang_vel.copy_(ang_vel)
    prev_ang_acc.copy_(ang_acc)
    return torch.sum(torch.square(ang_jerk), dim=1)
