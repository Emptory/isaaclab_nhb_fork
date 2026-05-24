from __future__ import annotations
import math
import torch
from typing import TYPE_CHECKING, Literal
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs.mdp.events import _randomize_prop_by_op
import random

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def randomize_rigid_body_inertia(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def set_rigid_object_relative_to_robot(
    env,
    env_ids: torch.Tensor,
    base_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    relative_pose: dict[str, float | tuple[float, float]],
    relative_velocity: dict[str, float],
) -> torch.Tensor:
    """Reset a rigid object to a pose expressed in a robot/body local frame."""
    base_asset: Articulation | RigidObject = env.scene[base_asset_cfg.name]
    target_object: RigidObject = env.scene[target_asset_cfg.name]

    base_asset_cfg.resolve(env.scene)
    target_asset_cfg.resolve(env.scene)

    if isinstance(base_asset, Articulation):
        body_id = base_asset_cfg.body_ids[0]
        base_pos = base_asset.data.body_link_pos_w[env_ids, body_id]
        base_quat = base_asset.data.body_link_quat_w[env_ids, body_id]
    else:
        base_pos = base_asset.data.root_pos_w[env_ids]
        base_quat = base_asset.data.root_quat_w[env_ids]

    num_resets = len(env_ids)

    def _sample_or_fill(key: str) -> torch.Tensor:
        value = relative_pose[key]
        if isinstance(value, tuple):
            return torch.rand(num_resets, device=env.device) * (value[1] - value[0]) + value[0]
        return torch.full((num_resets,), float(value), device=env.device)

    rel_pos_local = torch.stack([_sample_or_fill("x"), _sample_or_fill("y"), _sample_or_fill("z")], dim=1)
    object_pos = base_pos + math_utils.quat_apply(base_quat, rel_pos_local)

    rel_roll = _sample_or_fill("roll") if "roll" in relative_pose else torch.zeros(num_resets, device=env.device)
    rel_pitch = _sample_or_fill("pitch") if "pitch" in relative_pose else torch.zeros(num_resets, device=env.device)
    rel_yaw = _sample_or_fill("yaw") if "yaw" in relative_pose else torch.zeros(num_resets, device=env.device)
    object_quat = math_utils.quat_mul(base_quat, math_utils.quat_from_euler_xyz(rel_roll, rel_pitch, rel_yaw))

    rel_lin_vel_local = torch.stack(
        [
            torch.full((num_resets,), float(relative_velocity["x"]), device=env.device),
            torch.full((num_resets,), float(relative_velocity["y"]), device=env.device),
            torch.full((num_resets,), float(relative_velocity["z"]), device=env.device),
        ],
        dim=1,
    )
    rel_ang_vel_local = torch.stack(
        [
            torch.full((num_resets,), float(relative_velocity["roll"]), device=env.device),
            torch.full((num_resets,), float(relative_velocity["pitch"]), device=env.device),
            torch.full((num_resets,), float(relative_velocity["yaw"]), device=env.device),
        ],
        dim=1,
    )

    object_root_state = target_object.data.default_root_state[env_ids].clone()
    object_root_state[:, :3] = object_pos
    object_root_state[:, 3:7] = object_quat
    object_root_state[:, 7:10] = math_utils.quat_apply(base_quat, rel_lin_vel_local)
    object_root_state[:, 10:13] = math_utils.quat_apply(base_quat, rel_ang_vel_local)
    target_object.write_root_state_to_sim(object_root_state, env_ids)
    return rel_pos_local


def create_fixed_joint_to_body(
    env,
    env_ids: torch.Tensor | None,
    parent_body_path: str,
    child_body_path: str,
    joint_name: str,
    parent_local_pos: tuple[float, float, float],
    parent_local_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    child_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    child_local_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    """Create fixed joints between two prims under every selected environment."""
    from pxr import Gf, Sdf, UsdPhysics

    if env_ids is None:
        env_ids = range(env.scene.num_envs)
    elif isinstance(env_ids, slice):
        env_ids = range(env.scene.num_envs)
    elif isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().tolist()

    stage = env.scene.stage

    def _quatf(quat_wxyz: tuple[float, float, float, float]) -> Gf.Quatf:
        return Gf.Quatf(
            float(quat_wxyz[0]),
            Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
        )

    for env_id in env_ids:
        env_path = env.scene.env_prim_paths[int(env_id)]
        parent_path = f"{env_path}/{parent_body_path}"
        child_path = f"{env_path}/{child_body_path}"
        joint_path = f"{env_path}/{joint_name}"

        if stage.GetPrimAtPath(joint_path).IsValid():
            continue

        parent_prim = stage.GetPrimAtPath(parent_path)
        child_prim = stage.GetPrimAtPath(child_path)
        if not parent_prim.IsValid():
            raise ValueError(f"Fixed joint parent prim does not exist: {parent_path}")
        if not child_prim.IsValid():
            raise ValueError(f"Fixed joint child prim does not exist: {child_path}")

        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*parent_local_pos))
        joint.CreateLocalRot0Attr().Set(_quatf(parent_local_rot))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_local_pos))
        joint.CreateLocalRot1Attr().Set(_quatf(child_local_rot))
        joint.CreateBreakForceAttr().Set(3.4028234663852886e38)
        joint.CreateBreakTorqueAttr().Set(3.4028234663852886e38)


def random_joints_target_position_by_scale(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    在训练过程中随机设置某些关节的目标位置，模拟干扰
    注意,随机化的关节不能被action控制,否则会导致冲突
    例如：当只训练人形下半身时，可以随机化上半身关节的位置

    Args:
        env (ManagerBasedEnv): _description_
        env_ids (torch.Tensor): _description_
        position_range (tuple[float, float]): 随机化范围，默认关节角度的百分比
        asset_cfg (SceneEntityCfg, optional): 要被随机化的关节. Defaults to SceneEntityCfg("robot").
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # 获取采样上下限
    joint_pos_limits = asset.data.default_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids].clone()

    # 得到随机系数
    scale_low, scale_high = position_range
    scale = scale_low + (scale_high - scale_low) * torch.rand_like(asset.data.joint_pos[iter_env_ids, asset_cfg.joint_ids].clone())

    # 计算采样上下限
    low = joint_pos_limits[..., 0] * scale
    high = joint_pos_limits[..., 1] * scale

    # 采样随机化关节目标角度
    tar_joint_pos = low + (high - low) * torch.rand_like(low)

    # 设置关节目标角度
    asset.set_joint_position_target(tar_joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def randomize_joint_default_pos_add(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    default_pose_add_limit: tuple[float, float],
):
    """随机化关节默认位置,模拟实物中零点标定不准确。直接给关节默认位置加上噪声

    Args:
        env (ManagerBasedEnv): _description_
        env_ids (torch.Tensor): _description_
        asset_cfg (SceneEntityCfg): _description_
        default_pose_add_limit (tuple[float, float]): 随机化的范围
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    raw_default_pose = asset.data.default_joint_pos[env_ids].clone().squeeze()
    low, high = default_pose_add_limit
    noise = (torch.rand_like(raw_default_pose) * (high - low) + low).squeeze()

    asset.data.default_joint_pos = raw_default_pose + noise


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    # 注意：这行代码用于保存原始值，以便导出或重置时使用
    if not hasattr(asset.data, "default_joint_pos_nominal"):
         asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        # 这一步很关键，更新动作管理器的 offset，否则 action 还是基于原来的 offset 计算
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos
