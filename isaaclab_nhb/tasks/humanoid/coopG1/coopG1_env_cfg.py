from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_nhb.tasks.humanoid.G1.G1_asset_cfg import G1_29DOF_ACTION_SCALE, G1_29DOF_JOINT_ORDER

from . import coopG1_mdp
from .coopG1_actions import EMAJointPositionActionCfg
from .coopG1_scene_cfg import get_coop_scene_cfg


@configclass
class CoopG1ObservationsCfg:
    """协作搬运任务观测配置。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测。"""

        robot_0_base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot_0")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        robot_1_base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot_1")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        robot_0_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot_0")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        robot_1_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot_1")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        robot_0_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_0", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        robot_1_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        robot_0_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_0", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        robot_1_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class CriticCfg(ObsGroup):
        """价值网络观测。"""

        robot_0_base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        robot_1_base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        robot_0_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        robot_1_base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        robot_0_projected_gravity = ObsTerm(
            func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}
        )
        robot_1_projected_gravity = ObsTerm(
            func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}
        )
        robot_0_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_0", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
        )
        robot_1_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
        )
        robot_0_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_0", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
        )
        robot_1_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot_1", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True),
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CoopG1CommandsCfg:
    """Velocity command specifications."""

    base_velocity = coopG1_mdp.BoxLinearVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        ranges=coopG1_mdp.BoxLinearVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.3, 0.3),
            lin_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class CoopG1ActionsCfg:
    """双机器人关节位置动作配置。"""

    robot_0_joint_pos = EMAJointPositionActionCfg(
        asset_name="robot_0",
        joint_names=G1_29DOF_JOINT_ORDER,
        scale=G1_29DOF_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        alpha=0.15,
    )
    robot_1_joint_pos = EMAJointPositionActionCfg(
        asset_name="robot_1",
        joint_names=G1_29DOF_JOINT_ORDER,
        scale=G1_29DOF_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        alpha=0.15,
    )


@configclass
class CoopG1RewardsCfg:
    """协作任务的基础奖励项。"""

    track_box_lin_vel_xyz = RewTerm(
        func=coopG1_mdp.track_box_lin_vel_xyz_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    box_edge_projection = RewTerm(
        func=coopG1_mdp.box_edge_projection_exp,
        weight=2.0,
        params={"edge_axis": "x", "edge_length": 0.4, "target_length": 0.4, "std": 0.05},
    )
    box_height = RewTerm(func=coopG1_mdp.box_height_l2, weight=-2.0, params={"target_height": 0.95})
    box_up_axis = RewTerm(func=coopG1_mdp.box_up_axis_l2, weight=-2.0)
    box_linear_acc = RewTerm(func=coopG1_mdp.box_linear_acc_l2, weight=-0.02)
    box_angular_acc = RewTerm(func=coopG1_mdp.box_angular_acc_l2, weight=-0.01)
    box_linear_jerk = RewTerm(func=coopG1_mdp.box_linear_jerk_l2, weight=-2.0e-5)
    box_angular_jerk = RewTerm(func=coopG1_mdp.box_angular_jerk_l2, weight=-1.0e-5)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    robot_0_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)},
    )
    robot_1_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)},
    )
    robot_0_joint_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)},
    )
    robot_1_joint_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class CoopG1TerminationsCfg:
    """协作任务终止条件。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact_0 = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces_0",
                body_names=[
                    "pelvis",
                ],
            ),
            "threshold": 10.0,
        },
    )
    robot_0_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35, "asset_cfg": SceneEntityCfg("robot_0")},
    )
    illegal_contact_1 = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces_1",
                body_names=[
                    "pelvis",
                ],
            ),
            "threshold": 10.0,
        },
    )
    robot_1_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35, "asset_cfg": SceneEntityCfg("robot_1")},
    )


def _robot_asset_cfg(robot_index: int) -> SceneEntityCfg:
    return SceneEntityCfg(
        f"robot_{robot_index}",
        joint_names=G1_29DOF_JOINT_ORDER,
        preserve_order=True,
    )


def _make_observations_cfg(num_robots: int):
    @configclass
    class PolicyCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class CriticCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    for i in range(num_robots):
        robot_name = f"robot_{i}"
        action_name = f"robot_{i}_joint_pos"

        setattr(
            PolicyCfg,
            f"{robot_name}_velocity_commands",
            ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
        )
        setattr(
            PolicyCfg,
            f"{robot_name}_base_ang_vel",
            ObsTerm(
                func=mdp.base_ang_vel,
                params={"asset_cfg": SceneEntityCfg(robot_name)},
                noise=Unoise(n_min=-0.03, n_max=0.03),
            ),
        )
        setattr(
            PolicyCfg,
            f"{robot_name}_projected_gravity",
            ObsTerm(
                func=mdp.projected_gravity,
                params={"asset_cfg": SceneEntityCfg(robot_name)},
                noise=Unoise(n_min=-0.03, n_max=0.03),
            ),
        )
        setattr(
            PolicyCfg,
            f"{robot_name}_joint_pos",
            ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": _robot_asset_cfg(i)},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            ),
        )
        setattr(
            PolicyCfg,
            f"{robot_name}_joint_vel",
            ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": _robot_asset_cfg(i)},
                noise=Unoise(n_min=-1.5, n_max=1.5),
            ),
        )
        setattr(
            PolicyCfg,
            f"{robot_name}_actions",
            ObsTerm(func=mdp.last_action, params={"action_name": action_name}),
        )

        setattr(
            CriticCfg,
            f"{robot_name}_velocity_commands",
            ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
        )
        setattr(
            CriticCfg,
            f"{robot_name}_base_lin_vel",
            ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg(robot_name)}),
        )
        setattr(
            CriticCfg,
            f"{robot_name}_base_ang_vel",
            ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg(robot_name)}),
        )
        setattr(
            CriticCfg,
            f"{robot_name}_projected_gravity",
            ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg(robot_name)}),
        )
        setattr(
            CriticCfg,
            f"{robot_name}_joint_pos",
            ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _robot_asset_cfg(i)}),
        )
        setattr(
            CriticCfg,
            f"{robot_name}_joint_vel",
            ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _robot_asset_cfg(i)}),
        )
        setattr(
            CriticCfg,
            f"{robot_name}_actions",
            ObsTerm(func=mdp.last_action, params={"action_name": action_name}),
        )

    @configclass
    class DynamicCoopG1ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()
        critic: CriticCfg = CriticCfg()

    cfg = DynamicCoopG1ObservationsCfg()
    for name, value in PolicyCfg.__dict__.items():
        if name.startswith("robot_"):
            setattr(cfg.policy, name, value)
    for name, value in CriticCfg.__dict__.items():
        if name.startswith("robot_"):
            setattr(cfg.critic, name, value)
    return cfg


def _make_actions_cfg(num_robots: int):
    @configclass
    class DynamicCoopG1ActionsCfg:
        pass

    for i in range(num_robots):
        setattr(
            DynamicCoopG1ActionsCfg,
            f"robot_{i}_joint_pos",
            EMAJointPositionActionCfg(
                asset_name=f"robot_{i}",
                joint_names=G1_29DOF_JOINT_ORDER,
                scale=G1_29DOF_ACTION_SCALE,
                use_default_offset=True,
                preserve_order=True,
                alpha=0.15,
            ),
        )

    cfg = DynamicCoopG1ActionsCfg()
    for name, value in DynamicCoopG1ActionsCfg.__dict__.items():
        if name.startswith("robot_"):
            setattr(cfg, name, value)
    return cfg


def _make_rewards_cfg(num_robots: int):
    @configclass
    class DynamicCoopG1RewardsCfg:
        track_box_lin_vel_xyz = RewTerm(
            func=coopG1_mdp.track_box_lin_vel_xyz_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        box_edge_projection = RewTerm(
            func=coopG1_mdp.box_edge_projection_exp,
            weight=2.0,
            params={"edge_axis": "x", "edge_length": 0.4, "target_length": 0.4, "std": 0.05},
        )
        box_height = RewTerm(
            func=coopG1_mdp.box_height_l2,
            weight=-2.0,
            params={"target_height": 0.95},
        )
        box_up_axis = RewTerm(func=coopG1_mdp.box_up_axis_l2, weight=-2.0)
        box_linear_acc = RewTerm(func=coopG1_mdp.box_linear_acc_l2, weight=-0.02)
        box_angular_acc = RewTerm(func=coopG1_mdp.box_angular_acc_l2, weight=-0.01)
        box_linear_jerk = RewTerm(func=coopG1_mdp.box_linear_jerk_l2, weight=-2.0e-5)
        box_angular_jerk = RewTerm(func=coopG1_mdp.box_angular_jerk_l2, weight=-1.0e-5)

        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
        termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    for i in range(num_robots):
        robot_name = f"robot_{i}"
        setattr(
            DynamicCoopG1RewardsCfg,
            f"{robot_name}_flat_orientation",
            RewTerm(func=mdp.flat_orientation_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg(robot_name)}),
        )
        setattr(
            DynamicCoopG1RewardsCfg,
            f"{robot_name}_base_height",
            RewTerm(
                func=mdp.base_height_l2,
                weight=-1.0,
                params={"target_height": 0.75, "asset_cfg": SceneEntityCfg(robot_name)},
            ),
        )
        setattr(
            DynamicCoopG1RewardsCfg,
            f"{robot_name}_joint_pos_limits",
            RewTerm(func=mdp.joint_pos_limits, weight=-0.25, params={"asset_cfg": _robot_asset_cfg(i)}),
        )
        setattr(
            DynamicCoopG1RewardsCfg,
            f"{robot_name}_joint_torques_l2",
            RewTerm(func=mdp.joint_torques_l2, weight=-7.0e-5, params={"asset_cfg": _robot_asset_cfg(i)}),
        )
        setattr(
            DynamicCoopG1RewardsCfg,
            f"{robot_name}_joint_torque_limits",
            RewTerm(func=mdp.applied_torque_limits, weight=-0.1, params={"asset_cfg": _robot_asset_cfg(i)}),
        )

    cfg = DynamicCoopG1RewardsCfg()
    for name, value in DynamicCoopG1RewardsCfg.__dict__.items():
        if name.startswith(("track_", "box_", "action_", "termination_", "robot_")):
            setattr(cfg, name, value)
    return cfg


def _make_terminations_cfg(num_robots: int):
    @configclass
    class DynamicCoopG1TerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    for i in range(num_robots):
        setattr(
            DynamicCoopG1TerminationsCfg,
            f"illegal_contact_{i}",
            DoneTerm(
                func=mdp.illegal_contact,
                params={
                    "sensor_cfg": SceneEntityCfg(
                        f"contact_forces_{i}",
                        body_names=[
                            "pelvis",
                        ],
                    ),
                    "threshold": 10.0,
                },
            ),
        )
        setattr(
            DynamicCoopG1TerminationsCfg,
            f"robot_{i}_base_height",
            DoneTerm(
                func=mdp.root_height_below_minimum,
                params={"minimum_height": 0.35, "asset_cfg": SceneEntityCfg(f"robot_{i}")},
            ),
        )

    cfg = DynamicCoopG1TerminationsCfg()
    for name, value in DynamicCoopG1TerminationsCfg.__dict__.items():
        if name == "time_out" or name.startswith(("illegal_contact_", "robot_")):
            setattr(cfg, name, value)
    return cfg


@configclass
class CoopG1EnvCfg(ManagerBasedRLEnvCfg):
    """协作搬运木箱主环境配置。"""

    # 第一阶段默认只训练单机器人；多机器人后续通过 env.num_robots=N 显式开启
    num_robots: int = 1
    include_box: bool = True

    observations: CoopG1ObservationsCfg = CoopG1ObservationsCfg()
    commands: CoopG1CommandsCfg = CoopG1CommandsCfg()
    actions: CoopG1ActionsCfg = CoopG1ActionsCfg()
    rewards: CoopG1RewardsCfg = CoopG1RewardsCfg()
    terminations: CoopG1TerminationsCfg = CoopG1TerminationsCfg()

    def rebuild_dynamic_cfg(self):
        if self.num_robots < 1:
            raise ValueError(f"num_robots must be >= 1, got {self.num_robots}.")

        scene_num_envs = getattr(getattr(self, "scene", None), "num_envs", 2)
        scene_env_spacing = getattr(getattr(self, "scene", None), "env_spacing", 2.5)

        self.observations = _make_observations_cfg(self.num_robots)
        self.actions = _make_actions_cfg(self.num_robots)
        self.rewards = _make_rewards_cfg(self.num_robots)
        self.terminations = _make_terminations_cfg(self.num_robots)

        # 场景：延迟实例化，避免 import 阶段触发重依赖与副作用
        SceneCfg = get_coop_scene_cfg(num_robots=self.num_robots, include_box=self.include_box)
        self.scene = SceneCfg(num_envs=scene_num_envs, env_spacing=scene_env_spacing)

    def __post_init__(self):
        # 基本仿真参数
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005

        self.rebuild_dynamic_cfg()

        super().__post_init__()
