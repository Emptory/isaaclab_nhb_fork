from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb

from ..coopG1S0.coopG1S0_env_cfg import CoopG1S0FlatEnvCfg
from ..coopG1S1.coopG1S1_env_cfg import (
    CoopG1S1CommandsCfg,
    CoopG1S1EventCfg,
    CoopG1S1LegacyObsCfg,
    CoopG1S1ObsCfg,
    CoopG1S1RewardsCfg,
    CoopG1S1SceneCfg,
    CoopG1S1TerminationsCfg,
)


FIXED_PAYLOAD_REL_POS = (0.30, 0.0, 0.13)
FIXED_PAYLOAD_SIZE = (0.18, 0.55, 0.16)
HAND_ANCHOR_OFFSETS = (
    (0.05361310808, -0.00295905240, 0.00215413091),
    (0.05361310808, 0.00295905240, 0.00215413091),
)
TWO_HAND_REFERENCE_CSV = str(
    Path(__file__).resolve().parents[3] / "stimulationData/box_6dof_two_hand_19x2.csv"
)


@configclass
class CoopG1S2SceneCfg(CoopG1S1SceneCfg):
    """S1 robot scene with a dummy payload kept for S2 observation/reward compatibility."""

    hold_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoldBox",
        spawn=sim_utils.CuboidCfg(
            size=FIXED_PAYLOAD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.25, 0.1),
                metallic=0.1,
                opacity=0.08,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.72)),
    )


@configclass
class CoopG1S2ObsCfg:
    """Separate frozen-base and residual observations; all history is flattened directly into MLPs."""

    @configclass
    class BasePolicyCfg(CoopG1S1ObsCfg.PolicyCfg):
        pass

    @configclass
    class ResidualPolicyCfg(ObsGroup):
        """Only residual context: current upper command plus payload/grasp history."""

        hand_reference = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_reference"})
        payload_rel_pos = ObsTerm(
            func=mdp_nhb.asset_rel_pos,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            history_length=5,
        )
        payload_rel_quat = ObsTerm(
            func=mdp_nhb.asset_rel_quat,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            history_length=5,
        )
        payload_rel_lin_vel = ObsTerm(
            func=mdp_nhb.asset_rel_lin_vel,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            history_length=5,
        )
        payload_rel_ang_vel = ObsTerm(
            func=mdp_nhb.asset_rel_ang_vel,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            history_length=5,
        )
        payload_projected_gravity = ObsTerm(
            func=mdp_nhb.rigid_object_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("hold_box")},
            history_length=5,
        )
        hand_pos_in_payload = ObsTerm(
            func=mdp_nhb.body_rel_pos_to_object,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_rubber_hand", "right_rubber_hand"],
                    preserve_order=True,
                ),
                "object_cfg": SceneEntityCfg("hold_box"),
            },
            history_length=5,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(CoopG1S1ObsCfg.CriticCfg):
        hand_reference = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_reference"})
        payload_rel_pos = ObsTerm(
            func=mdp_nhb.asset_rel_pos,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
        )
        payload_rel_quat = ObsTerm(
            func=mdp_nhb.asset_rel_quat,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
        )
        payload_rel_lin_vel = ObsTerm(
            func=mdp_nhb.asset_rel_lin_vel,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
        )
        payload_rel_ang_vel = ObsTerm(
            func=mdp_nhb.asset_rel_ang_vel,
            params={
                "target_asset_cfg": SceneEntityCfg("hold_box"),
                "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
        )
        payload_projected_gravity = ObsTerm(
            func=mdp_nhb.rigid_object_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("hold_box")},
        )
        hand_pos_in_payload = ObsTerm(
            func=mdp_nhb.body_rel_pos_to_object,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_rubber_hand", "right_rubber_hand"],
                    preserve_order=True,
                ),
                "object_cfg": SceneEntityCfg("hold_box"),
            },
        )

    base_policy: BasePolicyCfg = BasePolicyCfg()
    residual_policy: ResidualPolicyCfg = ResidualPolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CoopG1S2LegacyObsCfg(CoopG1S2ObsCfg):
    """S2 observations with the original 515-D S1 observation for the frozen base actor."""

    @configclass
    class BasePolicyCfg(CoopG1S1LegacyObsCfg.PolicyCfg):
        pass

    base_policy: BasePolicyCfg = BasePolicyCfg()


@configclass
class CoopG1S2EventCfg(CoopG1S1EventCfg):
    pass


@configclass
class CoopG1S2CommandsCfg(CoopG1S1CommandsCfg):
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )
    hand_reference = mdp_nhb.TwoHandCsvReferenceCommandCfg(
        data_path=TWO_HAND_REFERENCE_CSV,
        dataset_dt=0.005,
        preview_steps=0,
        resampling_time_range=(20.0, 20.0),
        debug_vis=False,
    )


@configclass
class CoopG1S2RewardsCfg(CoopG1S1RewardsCfg):
    """S1 locomotion rewards plus command and payload stabilization objectives."""

    hand_payload_pose = RewTerm(
        func=mdp_nhb.two_hand_reference_position_exp,
        weight=4.0,
        params={
            "command_name": "hand_reference",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "hand_anchor_offsets": HAND_ANCHOR_OFFSETS,
            "std": 0.10,
        },
    )
    hand_target_orientation = RewTerm(
        func=mdp_nhb.two_hand_reference_orientation_exp,
        weight=1.0,
        params={
            "command_name": "hand_reference",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "hand_anchor_offsets": HAND_ANCHOR_OFFSETS,
            "std": 0.30,
        },
    )
    hand_target_velocity = RewTerm(
        func=mdp_nhb.two_hand_reference_linear_velocity_exp,
        weight=0.5,
        params={
            "command_name": "hand_reference",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "hand_anchor_offsets": HAND_ANCHOR_OFFSETS,
            "std": 0.30,
        },
    )
    hand_target_angular_velocity = RewTerm(
        func=mdp_nhb.two_hand_reference_angular_velocity_exp,
        weight=0.25,
        params={
            "command_name": "hand_reference",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "hand_anchor_offsets": HAND_ANCHOR_OFFSETS,
            "std": 0.50,
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
        params={"object_cfg": SceneEntityCfg("hold_box"), "lambda_exp": 6.0},
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
        params={"object_cfg": SceneEntityCfg("hold_box"), "lambda_exp": 1.5},
    )
    payload_lin_vel_z = RewTerm(
        func=mdp_nhb.object_lin_vel_z_exp,
        weight=0.10,
        params={"object_cfg": SceneEntityCfg("hold_box"), "lambda_exp": 2.0},
    )


@configclass
class CoopG1S2TerminationsCfg(CoopG1S1TerminationsCfg):
    pass


@configclass
class CoopG1S2FixedPayloadEnvCfg(CoopG1S0FlatEnvCfg):
    """S2 payload task trained by a frozen S1 base actor plus residual MLP."""

    scene: CoopG1S2SceneCfg = CoopG1S2SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )
    observations: CoopG1S2ObsCfg = CoopG1S2ObsCfg()
    commands: CoopG1S2CommandsCfg = CoopG1S2CommandsCfg()
    events: CoopG1S2EventCfg = CoopG1S2EventCfg()
    rewards: CoopG1S2RewardsCfg = CoopG1S2RewardsCfg()
    terminations: CoopG1S2TerminationsCfg = CoopG1S2TerminationsCfg()


@configclass
class CoopG1S2LegacyFixedPayloadEnvCfg(CoopG1S2FixedPayloadEnvCfg):
    """S2 residual environment for frozen 515-D S1 actors."""

    observations: CoopG1S2LegacyObsCfg = CoopG1S2LegacyObsCfg()
