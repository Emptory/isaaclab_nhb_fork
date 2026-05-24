import torch
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
import math
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, MySceneCfg


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


import isaaclab_nhb 
if not isaaclab_nhb.HEADLESS_FLAG:
    from isaaclab_nhb.envs.ui.manager_debug_rl_window import ManagerDebugRLEnvWindow
    from isaaclab_nhb.envs.manager_debug_rl_env import ManagerDebugRLEnv


import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb
from isaaclab_nhb.terrains.config.rough import ROUGH_TERRAINS_SIMPLE_CFG

from .Pegasus_asset_cfg import PEGASUS_CFG, PEGASUS_JOINT_NAMES
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from dataclasses import MISSING

if not isaaclab_nhb.HEADLESS_FLAG:
    class PegasusDebugWindow(ManagerDebugRLEnvWindow):
        """新建一个窗口用于绘制变量曲线"""
        env: ManagerDebugRLEnv
        def __init__(
            self,
            env: ManagerDebugRLEnv,
            window_name="IsaacLab",
            debug_window_name="debug info",
        ):
            # 调用父类的构造函数
            super().__init__(env, window_name, debug_window_name)

        def _register_debug_plots(self):
            """注册需要的图表"""
            self.register_plot("toruqe", [
                "lb1", "lb2", "lb3",
                "lf1", "lf2", "lf3",
                "rb1", "rb2", "rb3",
                "rf1", "rf2", "rf3"
            ])
            self.register_plot("gait I", ["I_lf", "I_lb", "I_rf", "I_rb"])

        def _update_debug_data(self) -> dict:
            """更新调试数据"""
            data = {}
            
            # 关节力矩
            asset = self.env.scene["robot"]
            lb1_torque = asset.data.applied_torque[0,0]
            lb2_torque = asset.data.applied_torque[0,1]
            lb3_torque = asset.data.applied_torque[0,2]
            lf1_torque = asset.data.applied_torque[0,3]
            lf2_torque = asset.data.applied_torque[0,4]
            lf3_torque = asset.data.applied_torque[0,5]
            rb1_torque = asset.data.applied_torque[0,6]
            rb2_torque = asset.data.applied_torque[0,7]
            rb3_torque = asset.data.applied_torque[0,8]
            rf1_torque = asset.data.applied_torque[0,9]
            rf2_torque = asset.data.applied_torque[0,10]
            rf3_torque = asset.data.applied_torque[0,11]
            data["toruqe"] = torch.stack([
                lb1_torque, lb2_torque, lb3_torque,
                lf1_torque, lf2_torque, lf3_torque,
                rb1_torque, rb2_torque, rb3_torque,
                rf1_torque, rf2_torque, rf3_torque,
            ])

            # 相位指示器
            gait_rew_class: mdp_nhb.QuadrupedGaitReward = self.env.reward_manager.cfg.gait_rew.func
            I_lf = gait_rew_class.LF_I[0]
            I_lb = gait_rew_class.LB_I[0]
            I_rf = gait_rew_class.RF_I[0]
            I_rb = gait_rew_class.RB_I[0]
            data["gait I"] = torch.stack([I_lf, I_lb, I_rf, I_rb], dim=0)
            
            return data

            # # 正余弦命令
            # gait_command = self.env.command_manager.get_command("gait_command")
            # sc_command = torch.stack(
            #     [gait_command[0,3],gait_command[0,4],gait_command[0,5],gait_command[0,6]]
            # )
            # self._add_var_to_dict("sc command", sc_command, ["sin_l","cos_l","sin_r","cos_r"])

@configclass
class PegasusRoughSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_SIMPLE_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.6,
            dynamic_friction=0.6,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = PEGASUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors 
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class PegasusRoughObsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # base角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        # base重力向量
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.0, n_max=0.0))
        # 线速度命令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=PEGASUS_JOINT_NAMES,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=PEGASUS_JOINT_NAMES,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-1.5, n_max=1.5))
        # 上一次的动作值
        actions = ObsTerm(func=mdp.last_action)
        # 正余弦信息
        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})

        # 在构造函数__init__后执行的“后构造函数”， 修改部分参数
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""
        # 机体线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        # base角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        # base重力向量
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.0, n_max=0.0))
        # 线速度命令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=PEGASUS_JOINT_NAMES,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.0, n_max=0.0))
        # 关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=PEGASUS_JOINT_NAMES,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.0, n_max=0.0))
        # 上一次的动作值
        actions = ObsTerm(func=mdp.last_action)
        # 正余弦信息
        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})
        # 高度采样 TODO:搞懂原理
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"),"offset": 0.0},
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    # 观测值配置组
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class PegasusRoughRewards:

    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=4.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # 机体平衡奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-5.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)

    # 运动平滑奖励
    action_smooth_tripple_l2 = RewTerm(func=mdp_nhb.action_smooth_tripple_l2,weight=-4e-6,params={"c_a":0.05, "c_da":2.5, "c_dda":1})
    dof_smooth_l2 = RewTerm(func=mdp_nhb.dof_smooth_l2,weight=-4e-6,params={"c_dof_v":0.02, "c_dof_a":0.025, "c_tor":1})
    body_smooth_l2 = RewTerm(func=mdp_nhb.body_smooth_l2,weight=-4e-6,
        params={
            "c_body_a":10, 
            "c_feet_a":1, 
            "base_asset_cfg":SceneEntityCfg("robot", body_names="base_link"),
            "feet_asset_cfg":SceneEntityCfg("robot", body_names=".*4_Link"),
        }
    )

    # 安全奖励
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    joint_vel_soft_limits = RewTerm(func=mdp_nhb.joint_vel_soft_limits, weight=-0.2, params={"soft_ratio":0.95})
    joint_tor_soft_limits = RewTerm(func=mdp_nhb.joint_tor_soft_limits, weight=-0.2, params={"soft_ratio":0.95})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 步态奖励
    base_height = RewTerm(func=mdp.base_height_l2,weight=-1.0,params={"target_height": 0.4})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*4_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*4_Link"),
        },
    )
    stand_still = RewTerm(
        func=mdp_nhb.stand_still_without_cmd,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*1_Joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*2_Joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*3_Joint"])},
    )
    
    gait_rew = RewTerm(
        func=mdp_nhb.QuadrupedGaitReward,
        weight=4.0,
        params={
            "lf_sensor_cfg": SceneEntityCfg("contact_forces", body_names="lf4_Link"),
            "rf_sensor_cfg": SceneEntityCfg("contact_forces", body_names="rf4_Link"),
            "lb_sensor_cfg": SceneEntityCfg("contact_forces", body_names="lb4_Link"),
            "rb_sensor_cfg": SceneEntityCfg("contact_forces", body_names="rb4_Link"),

            "lf_asset_cfg": SceneEntityCfg("robot", body_names="lf4_Link"),
            "rf_asset_cfg": SceneEntityCfg("robot", body_names="rf4_Link"),
            "lb_asset_cfg": SceneEntityCfg("robot", body_names="lb4_Link"),
            "rb_asset_cfg": SceneEntityCfg("robot", body_names="rb4_Link"),

            "foot_height_tar": 0.1
        }
    )

@configclass
class PegasusRoughEventCfg:
    """Configuration for events."""
    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.3, 1.0),
    #         "dynamic_friction_range": (0.3, 0.8),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )

@configclass
class PegasusRoughActionsCfg:
    """为MDP配置动作值"""

    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=PEGASUS_JOINT_NAMES, 
        scale=0.5, 
        use_default_offset=True,
        preserve_order=True)
    
@configclass
class PegasusRoughCommandsCfg:
    """Command specifications for the MDP."""

    # 速度跟踪命令
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
    )

    # 步态命令
    gait_command = mdp_nhb.QuadrupedGaitCommandCfg(
        resampling_time_range=(10, 10),
        ranges=mdp_nhb.QuadrupedGaitCommandCfg.Ranges(
            stance_rate=(0.5,0.5), 
            rf_offset=(0.5,0.5),
            lb_offset=(0.5,0.5),
            rb_offset=(0.0,0.0),
            gait_frequency=(1.25, 1.25)
        ),
    )     

@configclass
class PegasusRoughTerminationsCfg:
    """复位触发条件"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link",".*1_Link",".*2_Link"]), "threshold": 1.0},
    )

@configclass
class PegasusRoughCurriculumCfg:
    """课程"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)



@configclass
class PegasusRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: PegasusRoughSceneCfg = PegasusRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: PegasusRoughObsCfg = PegasusRoughObsCfg()
    actions: PegasusRoughActionsCfg = PegasusRoughActionsCfg()
    commands: PegasusRoughCommandsCfg = PegasusRoughCommandsCfg()
    # MDP settings
    rewards: PegasusRoughRewards = PegasusRoughRewards()
    terminations: PegasusRoughTerminationsCfg = PegasusRoughTerminationsCfg()
    events: PegasusRoughEventCfg = PegasusRoughEventCfg()
    curriculum: PegasusRoughCurriculumCfg = PegasusRoughCurriculumCfg()

    ui_window_class_type = ManagerBasedRLEnvWindow

    def __post_init__(self):
        super().__post_init__()
        # 仿真参数设置
        self.decimation = 4 # 算法50Hz
        self.sim.dt = 0.005 # 仿真200Hz
        self.episode_length_s = 20.0 # 1个episode20s


@configclass
class PegasusRoughEnvCfg_PLAY(PegasusRoughEnvCfg):

    # 开启图形渲染时才使用绘图窗口
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type:PegasusDebugWindow = PegasusDebugWindow

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
