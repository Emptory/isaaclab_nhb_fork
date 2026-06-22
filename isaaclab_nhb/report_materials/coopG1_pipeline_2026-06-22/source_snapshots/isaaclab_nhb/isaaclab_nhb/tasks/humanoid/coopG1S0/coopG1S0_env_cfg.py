import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb


from isaaclab_nhb.tasks.humanoid.G1.G1_asset_cfg import (
    G1_29DOF_CFG,
    G1_29DOF_JOINT_ORDER,
    G1_29DOF_ACTION_SCALE,
)

G1_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_SWING_FOOT_HEIGHT_M = 0.10

@configclass
class CoopG1S0SceneCfg(InteractiveSceneCfg):
    """Scene for single G1 locomotion."""

    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

@configclass
class CoopG1S0ObsCfg:
    """Observation config for single G1 locomotion."""

    @configclass
    class PolicyCfg(ObsGroup):

        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_29DOF_JOINT_ORDER,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_29DOF_JOINT_ORDER,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):

        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_29DOF_JOINT_ORDER,
                    preserve_order=True,
                )
            },
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_29DOF_JOINT_ORDER,
                    preserve_order=True,
                )
            },
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True
             
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class CoopG1S0ActionsCfg:
    """Action config for single G1 locomotion."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_JOINT_ORDER,
        scale=G1_29DOF_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={
            ".*": (-50.0, 50.0),
        },
    )
    
@configclass
class CoopG1S0CommandsCfg:
    """Velocity command config for single G1 locomotion."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.4),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )

    gait_command = mdp_nhb.BipedalGaitCommandCfg(
        resampling_time_range=(10.0, 10.0),
        ranges=mdp_nhb.BipedalGaitCommandCfg.Ranges(
            stance_rate=(0.65, 0.65),
            bipedal_offset=(0.5, 0.5),
            gait_frequency=(1.0, 1.0),
        ),
    )
    
@configclass
class CoopG1S0RewardsCfg:
    """Reward config for single G1 locomotion."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp_nhb.track_ang_vel_z_body_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    torso_lin_vel = RewTerm(
        func=mdp_nhb.body_lin_vel_z_exp,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="torso_link",
            ),
            "lambda_exp": 2.0,
        },
    )

    torso_ang_vel = RewTerm(
        func=mdp_nhb.body_ang_vel_xy_exp,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="torso_link",
            ),
            "lambda_exp": 1.0,
        },
    )

    upright_torso = RewTerm(
        func=mdp_nhb.body_upright_bonus_exp,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="torso_link",
            ),
            "lambda_exp": 4.0,
        },
    )

    torso_height = RewTerm(
        func=mdp_nhb.body_height_exp,
        weight=0.5,
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
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "lambda_exp": 1.0,
        },
    )

    upright_pelvis = RewTerm(
        func=mdp_nhb.body_upright_bonus_exp,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "lambda_exp": 4.0,
        },
    )

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    action_rate = RewTerm(
        func=mdp_nhb.action_rate_l2_clipped,
        weight=-0.05,
        params={
            "max_penalty": 100.0,
        },
    )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    energy = RewTerm(func=mdp_nhb.energy, weight=-2.0e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
        },
    )

    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names="waist_yaw_joint",
            ),
        },
    )

    torso_pelvis_yaw_alignment = RewTerm(
        func=mdp_nhb.body_body_yaw_alignment_exp,
        weight=2.0,
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

    gait_ensure = RewTerm(
        func=mdp_nhb.BipedalGaitEnsureReward,
        weight=1.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES[0]),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODY_NAMES[1]),
        },
    )

    feet_orientation = RewTerm(
        func=mdp_nhb.feet_yaw_alignment_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(G1_FOOT_BODY_NAMES),
                preserve_order=True,
            ),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "std": 0.5,
        },
    )

@configclass
class CoopG1S0TerminationsCfg:
    """Termination config for single G1 locomotion."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.4,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 1.0,
        },
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "pelvis",
                    "torso_link",
                ],
            ),
            "threshold": 5.0,
        },
    )

@configclass
class CoopG1S0EventCfg:
    """Event config for reset."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class CoopG1S0FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Flat locomotion env config for single G1."""

    scene: CoopG1S0SceneCfg = CoopG1S0SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )

    observations: CoopG1S0ObsCfg = CoopG1S0ObsCfg()
    actions: CoopG1S0ActionsCfg = CoopG1S0ActionsCfg()
    commands: CoopG1S0CommandsCfg = CoopG1S0CommandsCfg()
    rewards: CoopG1S0RewardsCfg = CoopG1S0RewardsCfg()
    terminations: CoopG1S0TerminationsCfg = CoopG1S0TerminationsCfg()
    events: CoopG1S0EventCfg = CoopG1S0EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
