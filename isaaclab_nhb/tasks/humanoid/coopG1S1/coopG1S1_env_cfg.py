from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb

from ..coopG1S0.coopG1S0_env_cfg import (
    CoopG1S0CommandsCfg,
    CoopG1S0EventCfg,
    CoopG1S0FlatEnvCfg,
    CoopG1S0ObsCfg,
    CoopG1S0RewardsCfg,
    CoopG1S0SceneCfg,
    CoopG1S0TerminationsCfg,
    G1_FOOT_BODY_NAMES,
    G1_SWING_FOOT_HEIGHT_M,
)


HOLD_BOX_REL_POS = (0.30, 0.0, 0.23)
HOLD_BOX_SIZE = (0.18, 0.55, 0.16)
HOLD_BOX_HALF_SIZE = tuple(size * 0.5 for size in HOLD_BOX_SIZE)

HOLD_HAND_TARGET_POS = (
    (0.350404, 0.220756, 0.086077),
    (0.350387, -0.220792, 0.086075),
)

HOLD_ARM_JOINT_POS = {
    "left_shoulder_pitch_joint": -0.60,
    "right_shoulder_pitch_joint": -0.60,
    "left_shoulder_roll_joint": 0.35,
    "right_shoulder_roll_joint": -0.35,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.70,
    "right_elbow_joint": 0.70,
    "left_wrist_roll_joint": -1.0,
    "right_wrist_roll_joint": 1.0,
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
class CoopG1S1ObsCfg(CoopG1S0ObsCfg):
    """S0 observations plus base linear velocity for the policy."""

    @configclass
    class PolicyCfg(CoopG1S0ObsCfg.PolicyCfg):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CoopG1S0ObsCfg.CriticCfg = CoopG1S0ObsCfg.CriticCfg()


@configclass
class CoopG1S1LegacyObsCfg(CoopG1S0ObsCfg):
    """Original 515-D S1 policy observation without base linear velocity."""

    pass


@configclass
class CoopG1S1EventCfg(CoopG1S0EventCfg):
    """S0 reset events only; S1 has no payload attachment."""

    pass


@configclass
class CoopG1S1CommandsCfg(CoopG1S0CommandsCfg):
    """Force S1 training commands to require forward walking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.4),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class CoopG1S1RewardsCfg(CoopG1S0RewardsCfg):
    """S0 locomotion rewards plus carry-pose shaping."""

    joint_deviation_arms = None

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": 0.35,
        },
    )

    track_lin_vel_xy_fine = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.10,
        },
    )

    # 已经在body坐标系下
    track_ang_vel_z = RewTerm(
        func=mdp_nhb.track_ang_vel_z_body_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    torso_ang_vel = RewTerm(
        func=mdp_nhb.body_ang_vel_xy_body_exp,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "lambda_exp": 1.0,
        },
    )

    upright_torso = RewTerm(
        func=mdp_nhb.body_upright_bonus_exp,
        weight=0.85,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "lambda_exp": 4.0,
        },
    )

    torso_height = RewTerm(
        func=mdp_nhb.body_height_exp,
        weight=1.0,
        params={
            "target_height": 0.82,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="torso_link",
            ),
            "lambda_exp": 10.0,
        },
    )

    pelvis_ang_vel = RewTerm(
        func=mdp_nhb.body_ang_vel_xy_body_exp,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "lambda_exp": 1.0,
        },
    )

    upright_pelvis = RewTerm(
        func=mdp_nhb.body_upright_bonus_exp,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "lambda_exp": 4.0,
        },
    )

    # URDF本质上可以修改为fixed
    torso_pelvis_yaw_alignment = RewTerm(
        func=mdp_nhb.body_body_yaw_alignment_exp,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="torso_link",
            ),
            "reference_asset_cfg": SceneEntityCfg(
                "robot",
                body_names="pelvis",
            ),
            "std": 0.3,
        },
    )


    torso_pelvis_gravity_alignment = None

    arm_target_pose = RewTerm(
        func=mdp_nhb.joint_target_l1,
        weight=-0.15,
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
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_positions": HOLD_HAND_TARGET_POS,
            "std": 0.10,
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
            "box_center": HOLD_BOX_REL_POS,
            "box_half_size": HOLD_BOX_HALF_SIZE,
            "margin": 0.0,
        },
    )

    feet_distance_y = RewTerm(
        func=mdp_nhb.biped_distance_y_l2,
        weight=-30.0,
        params={
            "min_distance": 0.21,
            "max_distance": 0.27,
            "command_name": "base_velocity",
            "velocity_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_ankle_roll_link",
                preserve_order=True,
            ),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_ankle_roll_link",
            ),
        },
    )
    foot_acc = RewTerm(
        func=mdp_nhb.foot_acc_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_ankle_roll_link",
            ),
        },
    )

    feet_orientation = RewTerm(
        func=mdp_nhb.feet_yaw_alignment_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_ankle_roll_link",
            ),
            "reference_asset_cfg": SceneEntityCfg(
                "robot",
                body_names="pelvis",
            ),
            "std": 0.5,
        },
    )

    ## 4) 足端与步态质量（步态自然性）
    # 步态相位与触地一致性奖励
    gait = RewTerm(
        func=mdp_nhb.BipedalGaitReward,
        weight=1.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES[0]),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES[1]),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES[0]),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODY_NAMES[1]),
            "foot_height_tar": G1_SWING_FOOT_HEIGHT_M,
        },
    )
    # 步态相位触地一致性约束（用于减少垫步）
    gait_ensure = RewTerm(
        func=mdp_nhb.BipedalGaitEnsureReward,
        weight=1.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES[0]),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES[1]),
        },
    )


@configclass
class CoopG1S1TerminationsCfg(CoopG1S0TerminationsCfg):
    """Stricter S1 terminations to reject crouched-but-stable local optima."""

    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.6,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 0.7,
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

    observations: CoopG1S1ObsCfg = CoopG1S1ObsCfg()
    commands: CoopG1S1CommandsCfg = CoopG1S1CommandsCfg()
    events: CoopG1S1EventCfg = CoopG1S1EventCfg()
    rewards: CoopG1S1RewardsCfg = CoopG1S1RewardsCfg()
    terminations: CoopG1S1TerminationsCfg = CoopG1S1TerminationsCfg()


@configclass
class CoopG1S1LegacyHoldBoxEnvCfg(CoopG1S1HoldBoxEnvCfg):
    """S1 environment compatible with pre-2026-06-22 515-D checkpoints."""

    observations: CoopG1S1LegacyObsCfg = CoopG1S1LegacyObsCfg()
