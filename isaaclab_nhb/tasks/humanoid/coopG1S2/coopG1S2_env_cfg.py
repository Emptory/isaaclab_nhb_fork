import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb

from ..coopG1S0.coopG1S0_env_cfg import (
    CoopG1S0EventCfg,
    CoopG1S0FlatEnvCfg,
    CoopG1S0ObsCfg,
    CoopG1S0RewardsCfg,
    CoopG1S0SceneCfg,
    CoopG1S0TerminationsCfg,
)


FIXED_PAYLOAD_REL_POS = (0.30, 0.0, 0.13)
FIXED_PAYLOAD_SIZE = (0.18, 0.55, 0.16)
FIXED_PAYLOAD_HALF_SIZE = tuple(size * 0.5 for size in FIXED_PAYLOAD_SIZE)

HAND_TARGET_POS = (
    (0.30, 0.32, 0.03),
    (0.30, -0.32, 0.03),
)

ARM_CARRY_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.30,
    "right_shoulder_pitch_joint": 0.30,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97,
    "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


@configclass
class CoopG1S2SceneCfg(CoopG1S0SceneCfg):
    """Scene for hand-attached payload locomotion."""

    hold_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoldBox",
        spawn=sim_utils.CuboidCfg(
            size=FIXED_PAYLOAD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.25, 0.1), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.72)),
    )


@configclass
class CoopG1S2EventCfg(CoopG1S0EventCfg):
    """S0 reset events; D6 hand-box attachments are created by CoopG1S2Env."""

    pass


@configclass
class CoopG1S2RewardsCfg(CoopG1S0RewardsCfg):
    """S0 locomotion rewards plus weak carry-pose shaping."""

    arm_target_pose = RewTerm(
        func=mdp_nhb.joint_target_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
            "targets": ARM_CARRY_JOINT_POS,
        },
    )

    hand_payload_pose = RewTerm(
        func=mdp_nhb.body_body_rel_pos_exp,
        weight=0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_positions": HAND_TARGET_POS,
            "std": 0.18,
        },
    )

    upper_body_box_overlap = RewTerm(
        func=mdp_nhb.body_inside_box_penalty,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_rubber_hand",
                    "right_rubber_hand",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                    "left_wrist_pitch_link",
                    "right_wrist_pitch_link",
                    "left_elbow_link",
                    "right_elbow_link",
                ],
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "box_center": FIXED_PAYLOAD_REL_POS,
            "box_half_size": FIXED_PAYLOAD_HALF_SIZE,
            "margin": 0.02,
        },
    )

    payload_rel_pose = RewTerm(
        func=mdp_nhb.object_rel_pos_exp,
        weight=0.25,
        params={
            "object_cfg": SceneEntityCfg("hold_box"),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_pos": FIXED_PAYLOAD_REL_POS,
            "std": 0.25,
        },
    )

    payload_upright = RewTerm(
        func=mdp_nhb.object_upright_bonus_exp,
        weight=0.40,
        params={
            "object_cfg": SceneEntityCfg("hold_box"),
            "lambda_exp": 6.0,
        },
    )

    payload_rel_rot = RewTerm(
        func=mdp_nhb.object_rel_rot_exp,
        weight=0.30,
        params={
            "object_cfg": SceneEntityCfg("hold_box"),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_quat": (1.0, 0.0, 0.0, 0.0),
            "std": 0.45,
        },
    )

    payload_ang_vel = RewTerm(
        func=mdp_nhb.object_ang_vel_exp,
        weight=0.20,
        params={
            "object_cfg": SceneEntityCfg("hold_box"),
            "lambda_exp": 1.5,
        },
    )

    payload_lin_vel_z = RewTerm(
        func=mdp_nhb.object_lin_vel_z_exp,
        weight=0.10,
        params={
            "object_cfg": SceneEntityCfg("hold_box"),
            "lambda_exp": 2.0,
        },
    )


@configclass
class CoopG1S2FixedPayloadEnvCfg(CoopG1S0FlatEnvCfg):
    """Single G1 hand-attached payload locomotion task."""

    scene: CoopG1S2SceneCfg = CoopG1S2SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )

    observations: CoopG1S0ObsCfg = CoopG1S0ObsCfg()
    events: CoopG1S2EventCfg = CoopG1S2EventCfg()
    rewards: CoopG1S2RewardsCfg = CoopG1S2RewardsCfg()
    terminations: CoopG1S0TerminationsCfg = CoopG1S0TerminationsCfg()
