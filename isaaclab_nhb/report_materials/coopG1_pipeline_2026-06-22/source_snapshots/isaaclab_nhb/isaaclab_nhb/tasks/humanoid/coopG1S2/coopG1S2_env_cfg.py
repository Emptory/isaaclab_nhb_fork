import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
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
    CoopG1S1ObsCfg,
    CoopG1S1RewardsCfg,
    CoopG1S1SceneCfg,
    CoopG1S1TerminationsCfg,
)


FIXED_PAYLOAD_REL_POS = (0.30, 0.0, 0.13)
FIXED_PAYLOAD_SIZE = (0.18, 0.55, 0.16)


@configclass
class CoopG1S2SceneCfg(CoopG1S1SceneCfg):
    """S1 robot scene with the hand-attached payload restored for residual training."""

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
class CoopG1S2ObsCfg:
    """Separate frozen-base and residual observations; all history is flattened directly into MLPs."""

    @configclass
    class BasePolicyCfg(CoopG1S1ObsCfg.PolicyCfg):
        pass

    @configclass
    class ResidualPolicyCfg(CoopG1S1ObsCfg.PolicyCfg):
        hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_target"})
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

    @configclass
    class CriticCfg(CoopG1S1ObsCfg.CriticCfg):
        hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_target"})
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
class CoopG1S2EventCfg(CoopG1S1EventCfg):
    pass


@configclass
class CoopG1S2CommandsCfg(CoopG1S1CommandsCfg):
    hand_target = mdp_nhb.EndEffectorTargetCommandCfg(
        resampling_time_range=(2.0, 4.0),
        debug_vis=False,
    )


@configclass
class CoopG1S2RewardsCfg(CoopG1S1RewardsCfg):
    """S1 locomotion rewards plus command and payload stabilization objectives."""

    hand_payload_pose = RewTerm(
        func=mdp_nhb.body_command_pos_exp,
        weight=4.0,
        params={
            "command_name": "hand_target",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "std": 0.10,
        },
    )
    hand_target_velocity = RewTerm(
        func=mdp_nhb.body_command_lin_vel_exp,
        weight=0.5,
        params={
            "command_name": "hand_target",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "std": 0.5,
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
