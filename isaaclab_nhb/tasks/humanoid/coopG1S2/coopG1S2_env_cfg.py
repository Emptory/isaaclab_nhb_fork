from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb

from ..G1.G1_asset_cfg import G1_29DOF_JOINT_ORDER
from ..G1.G1_asset_cfg import G1_29DOF_ACTION_SCALE
from ..coopG1S0.coopG1S0_env_cfg import CoopG1S0FlatEnvCfg
from ..coopG1S1.coopG1S1_env_cfg import (
    CoopG1S1CommandsCfg,
    CoopG1S1EventCfg,
    CoopG1S1LegacyObsCfg,
    CoopG1S1ObsCfg,
    CoopG1S1RewardsCfg,
    CoopG1S1SceneCfg,
    CoopG1S1TerminationsCfg,
    HOLD_ARM_JOINT_POS,
)


HAND_ANCHOR_OFFSETS = (
    (0.05361310808, -0.01795904764, 0.00216607581),
    (0.05361310808, 0.01795904764, 0.00214218601),
)
TWO_HAND_REFERENCE_CSV = str(
    Path(__file__).resolve().parents[3] / "stimulationData/box_6dof_two_hand_19x2.csv"
)
S2_ARM_JOINT_ORDER = G1_29DOF_JOINT_ORDER[15:]
S2_ARM_ACTION_INDICES = tuple(range(15, len(G1_29DOF_JOINT_ORDER)))
S2_ARM_RESIDUAL_SCALE = 0.5


@configclass
class VirtualTwoHandForceCfg:
    """Paper-style translational virtual springs for the two palm anchors.

    The paper uses ``F = Kp * (x_s - x_g) + Kd * (v_s - v_g)`` with
    ``Kp=700`` and ``Kd=6``. The mask below makes the current S2 task a
    translational hybrid: x/y retain trajectory rewards and z is controlled
    by force. Setting all entries to one recovers the paper's full 3-D force
    mode.

    The CSV wrench is hand-on-payload, whereas the spring is
    environment-on-hand; ``csv_force_is_hand_on_payload`` performs that
    Newton's-third-law sign conversion exactly once.
    """

    enabled: bool = True
    asset_name: str = "robot"
    reference_body_name: str = "torso_link"
    hand_body_names: tuple[str, str] = (
        "left_rubber_hand",
        "right_rubber_hand",
    )
    hand_anchor_offsets: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] = HAND_ANCHOR_OFFSETS
    force_control_axes: tuple[float, float, float] = (0.0, 0.0, 1.0)
    stiffness_n_per_m: tuple[float, float, float] = (700.0, 700.0, 700.0)
    damping_ns_per_m: tuple[float, float, float] = (6.0, 6.0, 6.0)
    max_force_n: tuple[float, float, float] = (20.0, 20.0, 20.0)
    csv_force_is_hand_on_payload: bool = True


@configclass
class CoopG1S2SceneCfg(CoopG1S1SceneCfg):
    """S1 robot scene used for kinematic two-hand trajectory tracking.

    The old translucent ``hold_box`` was a kinematic dummy that neither
    interacted with the robot nor followed the CSV.  Keeping it in policy
    observations therefore supplied a misleading constant state.  The
    payload path can still be rendered by play-time markers without adding a
    fake physics asset to the training state.
    """

    pass


@configclass
class CoopG1S2ActionsCfg:
    """Joint targets with a virtual-spring update at every physics substep."""

    joint_pos = mdp_nhb.VirtualSpringJointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_JOINT_ORDER,
        scale=G1_29DOF_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-50.0, 50.0)},
    )


@configclass
class CoopG1S2ObsCfg:
    """Frozen-S1, residual-actor, and critic observations with explicit semantics."""

    @configclass
    class BasePolicyCfg(CoopG1S1ObsCfg.PolicyCfg):
        # The frozen S1 actor was trained with its own previous action.  In S2,
        # mdp.last_action would instead expose base + residual and would change
        # the meaning of the otherwise shape-compatible 530-D observation.
        actions = ObsTerm(func=mdp_nhb.previous_base_action)

    @configclass
    class ResidualPolicyCfg(ObsGroup):
        """207-D closed-loop residual proprioception/reference history.

        Layout:
          26  current p/q/v/w reference for both hands
          120 five frames of directional p/R/v/w tracking error
          28  upper-body joint position and velocity
          12  projected gravity, base velocities, and velocity command
           7  gait command
          14  previously applied arm residual

        Force command/mode live in a separate 12-D group so they can be shared
        by actor and critic without inheriting the critic's five-frame history.
        """

        hand_reference = ObsTerm(
            func=mdp_nhb.two_hand_reference_kinematics,
            params={"command_name": "hand_reference"},
        )
        hand_reference_error = ObsTerm(
            func=mdp_nhb.two_hand_reference_tracking_error,
            params={"command_name": "hand_reference"},
            history_length=5,
        )
        upper_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=S2_ARM_JOINT_ORDER,
                    preserve_order=True,
                )
            },
        )
        upper_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=S2_ARM_JOINT_ORDER,
                    preserve_order=True,
                )
            },
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        gait_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "gait_command"},
        )
        previous_residual = ObsTerm(
            func=mdp_nhb.previous_residual_action,
            params={"action_indices": S2_ARM_ACTION_INDICES},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(CoopG1S1ObsCfg.CriticCfg):
        # The inherited group has five-frame history. Its 106-D S1 state plus
        # 26-D reference, 24-D error, and 14-D residual remains 5 * 170 = 850.
        actions = ObsTerm(func=mdp_nhb.previous_base_action)
        hand_reference = ObsTerm(
            func=mdp_nhb.two_hand_reference_kinematics,
            params={"command_name": "hand_reference"},
        )
        hand_reference_error = ObsTerm(
            func=mdp_nhb.two_hand_reference_tracking_error,
            params={"command_name": "hand_reference"},
        )
        previous_residual = ObsTerm(
            func=mdp_nhb.previous_residual_action,
            params={"action_indices": S2_ARM_ACTION_INDICES},
        )
    @configclass
    class ForceContextCfg(ObsGroup):
        """Deployable force command and hybrid-control mode, 12-D."""

        target_virtual_force = ObsTerm(func=mdp_nhb.two_hand_virtual_force_target)
        force_control_axes = ObsTerm(func=mdp_nhb.two_hand_force_control_axes)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ForcePrivilegedCfg(ObsGroup):
        """Exact virtual force used only as estimator supervision."""

        actual_virtual_force = ObsTerm(func=mdp_nhb.two_hand_actual_virtual_force)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    base_policy: BasePolicyCfg = BasePolicyCfg()
    residual_policy: ResidualPolicyCfg = ResidualPolicyCfg()
    critic: CriticCfg = CriticCfg()
    force_context: ForceContextCfg = ForceContextCfg()
    force_privileged: ForcePrivilegedCfg = ForcePrivilegedCfg()


@configclass
class CoopG1S2LegacyObsCfg(CoopG1S2ObsCfg):
    """S2 observations for the original 515-D frozen S1 actors."""

    @configclass
    class BasePolicyCfg(CoopG1S1LegacyObsCfg.PolicyCfg):
        actions = ObsTerm(func=mdp_nhb.previous_base_action)

    base_policy: BasePolicyCfg = BasePolicyCfg()


@configclass
class CoopG1S2EventCfg(CoopG1S1EventCfg):
    # Keep S1's observation zero point unchanged, but start the simulated
    # joints from its learned carry pose before aligning CSV time zero.
    reset_robot_joints = EventTerm(
        func=mdp_nhb.reset_joints_to_named_positions,
        mode="reset",
        params={"joint_positions": HOLD_ARM_JOINT_POS},
    )


@configclass
class CoopG1S2CommandsCfg(CoopG1S1CommandsCfg):
    # 0.30 m/s lies inside the frozen S1 training domain (0.25--0.40 m/s)
    # and must match the MATLAB/CSV trajectory generator.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.30, 0.30),
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
    """S1 locomotion rewards plus closed-loop two-hand trajectory tracking."""

    # These inherited terms would pull the arms back to the fixed S1 carry
    # pose or regularize base + residual together.  S2 instead regularizes the
    # residual component explicitly.
    arm_target_pose = None
    action_rate = None
    # Inherited from S1, this term assumes a torso-fixed box volume.  S2 has
    # no simulated payload yet, so retaining it would penalize an imaginary
    # object rather than trajectory tracking.
    upper_body_box_overlap = None

    # A dedicated zero-valued scheduling term records p/R/v/w metrics before
    # terminated environments are reset.  Keep the weight non-zero because
    # RewardManager skips zero-weight terms without calling their function.
    hand_tracking_metrics = RewTerm(
        func=mdp_nhb.record_two_hand_reference_metrics,
        weight=1.0,
        params={"command_name": "hand_reference"},
    )
    hand_force_metrics = RewTerm(
        func=mdp_nhb.record_two_hand_virtual_force_metrics,
        weight=1.0,
    )

    # The frozen S1 gait creates a deterministic 1 Hz disturbance at the
    # hands.  The residual already observes gait sin/cos, base motion, and
    # directional hand errors, so use a narrow kernel for precision plus a
    # wider kernel that still supplies gradient during the initial transient.
    hand_payload_pose = RewTerm(
        func=mdp_nhb.two_hand_hybrid_position_exp,
        weight=4.0,
        params={"std": 0.04},
    )
    hand_payload_pose_wide = RewTerm(
        func=mdp_nhb.two_hand_hybrid_position_exp,
        weight=2.0,
        params={"std": 0.20},
    )
    hand_target_orientation = RewTerm(
        func=mdp_nhb.two_hand_reference_orientation_exp,
        weight=2.0,
        params={
            "command_name": "hand_reference",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_rubber_hand", "right_rubber_hand"],
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "hand_anchor_offsets": HAND_ANCHOR_OFFSETS,
            "std": 0.15,
        },
    )
    hand_target_orientation_wide = RewTerm(
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
            "std": 0.60,
        },
    )
    hand_target_velocity = RewTerm(
        func=mdp_nhb.two_hand_hybrid_linear_velocity_exp,
        weight=2.0,
        params={"std": 0.10},
    )
    hand_target_velocity_wide = RewTerm(
        func=mdp_nhb.two_hand_hybrid_linear_velocity_exp,
        weight=0.5,
        params={"std": 0.35},
    )
    hand_target_angular_velocity = RewTerm(
        func=mdp_nhb.two_hand_reference_angular_velocity_exp,
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
            "std": 0.50,
        },
    )
    hand_target_angular_velocity_wide = RewTerm(
        func=mdp_nhb.two_hand_reference_angular_velocity_exp,
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
            "std": 1.50,
        },
    )
    hand_virtual_force = RewTerm(
        func=mdp_nhb.two_hand_virtual_force_tracking_exp,
        weight=5.0,
        params={"std": 0.25},
    )
    hand_virtual_force_wide = RewTerm(
        func=mdp_nhb.two_hand_virtual_force_tracking_exp,
        weight=1.0,
        params={"std": 1.00},
    )
    hand_virtual_force_clamp = RewTerm(
        func=mdp_nhb.two_hand_virtual_force_clamp_fraction,
        weight=-0.10,
    )
    residual_action_magnitude = RewTerm(
        func=mdp_nhb.residual_action_l2,
        weight=-0.01,
        params={"action_indices": S2_ARM_ACTION_INDICES},
    )
    residual_action_rate = RewTerm(
        func=mdp_nhb.residual_action_rate_l2,
        weight=-0.02,
        params={"action_indices": S2_ARM_ACTION_INDICES},
    )


@configclass
class CoopG1S2TerminationsCfg(CoopG1S1TerminationsCfg):
    pass


@configclass
class CoopG1S2FixedPayloadEnvCfg(CoopG1S0FlatEnvCfg):
    """S2 trajectory task controlled by a frozen S1 actor plus arm residual."""

    scene: CoopG1S2SceneCfg = CoopG1S2SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )
    actions: CoopG1S2ActionsCfg = CoopG1S2ActionsCfg()
    observations: CoopG1S2ObsCfg = CoopG1S2ObsCfg()
    commands: CoopG1S2CommandsCfg = CoopG1S2CommandsCfg()
    events: CoopG1S2EventCfg = CoopG1S2EventCfg()
    rewards: CoopG1S2RewardsCfg = CoopG1S2RewardsCfg()
    terminations: CoopG1S2TerminationsCfg = CoopG1S2TerminationsCfg()
    virtual_force: VirtualTwoHandForceCfg = VirtualTwoHandForceCfg()
    # Diagnostic copy of the active policy bound.  It does not alter action
    # processing; the agent config imports the same constant below.
    residual_action_limit: float = S2_ARM_RESIDUAL_SCALE


@configclass
class CoopG1S2LegacyFixedPayloadEnvCfg(CoopG1S2FixedPayloadEnvCfg):
    """S2 residual environment for frozen 515-D S1 actors."""

    observations: CoopG1S2LegacyObsCfg = CoopG1S2LegacyObsCfg()
