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


HOLD_BOX_REL_POS = (0.30, 0.0, 0.13)
HOLD_BOX_SIZE = (0.18, 0.55, 0.16)
HOLD_BOX_HALF_SIZE = tuple(size * 0.5 for size in HOLD_BOX_SIZE)

HOLD_HAND_TARGET_POS = (
    (0.39, 0.22, 0.03),
    (0.39, -0.22, 0.03),
)

HOLD_ARM_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.35,
    "right_shoulder_roll_joint": -0.35,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.05,
    "right_elbow_joint": 1.05,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


@configclass
class CoopG1S1SceneCfg(CoopG1S0SceneCfg):
    """Scene for single G1 locomotion without a payload asset."""

    pass


@configclass
class CoopG1S1EventCfg(CoopG1S0EventCfg):
    """S0 reset events only; S1 has no payload attachment."""

    pass


@configclass
class CoopG1S1RewardsCfg(CoopG1S0RewardsCfg):
    """S0 locomotion rewards plus carry-pose shaping."""

    arm_target_pose = RewTerm(
        func=mdp_nhb.joint_target_l1,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
            "targets": HOLD_ARM_JOINT_POS,
        },
    )

    hand_payload_pose = RewTerm(
        func=mdp_nhb.body_body_rel_pos_exp,
        weight=0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_positions": HOLD_HAND_TARGET_POS,
            "std": 0.14,
        },
    )

    upper_body_box_overlap = RewTerm(
        func=mdp_nhb.body_inside_box_penalty,
        weight=-0.05,
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
            "box_center": HOLD_BOX_REL_POS,
            "box_half_size": HOLD_BOX_HALF_SIZE,
            "margin": 0.0,
        },
    )


@configclass
class CoopG1S1HoldBoxEnvCfg(CoopG1S0FlatEnvCfg):
    """Single G1 locomotion task with carry-pose pretraining."""

    scene: CoopG1S1SceneCfg = CoopG1S1SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )

    observations: CoopG1S0ObsCfg = CoopG1S0ObsCfg()
    events: CoopG1S1EventCfg = CoopG1S1EventCfg()
    rewards: CoopG1S1RewardsCfg = CoopG1S1RewardsCfg()
    terminations: CoopG1S0TerminationsCfg = CoopG1S0TerminationsCfg()
