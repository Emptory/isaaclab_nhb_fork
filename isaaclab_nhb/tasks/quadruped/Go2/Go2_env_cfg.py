import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import isaaclab_nhb
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb

if not isaaclab_nhb.HEADLESS_FLAG:
    from isaaclab_nhb.envs.ui.manager_debug_rl_window import ManagerDebugRLEnvWindow
    from isaaclab_nhb.envs.manager_debug_rl_env import ManagerDebugRLEnv

from .Go2_asset_cfg import GO2_CFG, GO2_JOINT_NAMES

if not isaaclab_nhb.HEADLESS_FLAG:
    class Go2DebugWindow(ManagerDebugRLEnvWindow):
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
            pass

        def _update_debug_data(self) -> dict:
            """更新调试数据"""
            data = {}
            return data
            # 关节力矩
            # asset = self.env.scene["robot"]
            # lb1_torque = asset.data.applied_torque[0,0]
            # lb2_torque = asset.data.applied_torque[0,1]
            # lb3_torque = asset.data.applied_torque[0,2]
            # lf1_torque = asset.data.applied_torque[0,3]
            # lf2_torque = asset.data.applied_torque[0,4]
            # lf3_torque = asset.data.applied_torque[0,5]
            # rb1_torque = asset.data.applied_torque[0,6]
            # rb2_torque = asset.data.applied_torque[0,7]
            # rb3_torque = asset.data.applied_torque[0,8]
            # rf1_torque = asset.data.applied_torque[0,9]
            # rf2_torque = asset.data.applied_torque[0,10]
            # rf3_torque = asset.data.applied_torque[0,11]
            # torque = torch.stack([
            #     lb1_torque,lb2_torque,lb3_torque,
            #     lf1_torque,lf2_torque,lf3_torque,
            #     rb1_torque,rb2_torque,rb3_torque,
            #     rf1_torque,rf2_torque,rf3_torque,
            # ])
            # self._add_var_to_dict(
            #     "toruqe",
            #     torque,
            #     ["lb1","lb2","lb3",
            #     "lf1","lf2","lf3",
            #     "rb1","rb2","rb3",
            #     "rf1","rf2","rf3",], "plot")

            # # 相位指示器
            # gait_rew_class: mdp_nhb.QuadrupedGaitReward = self.env.reward_manager.cfg.gait_rew.func
            # I_lf = gait_rew_class.LF_I[0]
            # I_lb = gait_rew_class.LB_I[0]
            # I_rf = gait_rew_class.RF_I[0]
            # I_rb = gait_rew_class.RB_I[0]
            # gait_I = torch.stack([I_lf,I_lb,I_rf,I_rb],dim=0)
            # self._add_var_to_dict("gait I",gait_I,["I_lf","I_lb","I_rf","I_rb"], "plot")

            # # 正余弦命令
            # gait_command = self.env.command_manager.get_command("gait_command")
            # sc_command = torch.stack(
            #     [gait_command[0,3],gait_command[0,4],gait_command[0,5],gait_command[0,6]]
            # )
            # self._add_var_to_dict("sc command", sc_command, ["sin_l","cos_l","sin_r","cos_r"], "plot")

@configclass
class Go2RoughSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            sub_terrains={
                "boxes": ROUGH_TERRAINS_CFG.sub_terrains["boxes"].replace(grid_height_range=(0.025, 0.1)),
                "random_rough": ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].replace(
                    noise_range=(0.01, 0.06),
                    noise_step=0.01
                ),
            }
        ),
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,  # 与父类保持一致（UnitreeGo2没改这个）
            dynamic_friction=1.0,  # 与父类保持一致（UnitreeGo2没改这个）
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors 
    height_scanner = RayCasterCfg(  # 总的高程图
        prim_path="{ENV_REGEX_NS}/Robot/radar",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.55, 1.55], ordering="yx"),
        debug_vis=True,
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
class Go2RoughObsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # base角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        # base重力向量
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 线速度命令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=GO2_JOINT_NAMES,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=GO2_JOINT_NAMES,
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # base重力向量
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 线速度命令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=GO2_JOINT_NAMES,
                                                            preserve_order=True)})
        # 关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=GO2_JOINT_NAMES,
                                                            preserve_order=True)})
        # 上一次的动作值
        actions = ObsTerm(func=mdp.last_action)
        # 正余弦信息
        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"),"offset": 0.0},
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
class Go2RoughRewards:
    """Reward terms for the MDP."""

    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=5.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})

    # 机体平衡奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # 运动平滑奖励
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smooth_l2 = RewTerm(func=mdp_nhb.action_smooth_l2, weight=-0.01)

    # 安全奖励
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)
    joint_vel_soft_limits = RewTerm(func=mdp_nhb.joint_vel_soft_limits, weight=-1.0, params={"soft_ratio":0.9})
    joint_tor_soft_limits = RewTerm(func=mdp_nhb.joint_tor_soft_limits, weight=-1.0, params={"soft_ratio":0.9})
    joint_torque_limits = RewTerm(func=mdp.applied_torque_limits,weight=-0.2)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    non_foot_collision_penalty = RewTerm(
        func=mdp.illegal_contact,
        weight=-10.0,  # 惩罚权重，可根据需求调整
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*foot$).*"),  # 匹配非足部的部件（排除以foot结尾的）
            "threshold": 1.0,  # 碰撞力阈值
        },
    )


    # 步态奖励
    base_height = RewTerm(func=mdp.base_height_l2,weight=-1.0,params={"target_height": 0.34})  # Go2初始高度0.38，目标稍低保持稳定
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    stand_still_joint_pos = RewTerm(
        func=mdp_nhb.stand_still_without_cmd,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint.*")},
    )
    gait_rew = RewTerm(
        func=mdp_nhb.QuadrupedGaitReward,
        weight=4.0,
        params={
            "lf_sensor_cfg": SceneEntityCfg("contact_forces", body_names="FL_foot"),
            "rf_sensor_cfg": SceneEntityCfg("contact_forces", body_names="FR_foot"),
            "lb_sensor_cfg": SceneEntityCfg("contact_forces", body_names="RL_foot"),
            "rb_sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot"),

            "lf_asset_cfg": SceneEntityCfg("robot", body_names="FL_foot"),
            "rf_asset_cfg": SceneEntityCfg("robot", body_names="FR_foot"),
            "lb_asset_cfg": SceneEntityCfg("robot", body_names="RL_foot"),
            "rb_asset_cfg": SceneEntityCfg("robot", body_names="RR_foot"),

            "foot_height_tar": 0.1
        }
    )


@configclass
class Go2RoughEventCfg:
    """Configuration for events."""
    # startup
    # randomize_rigid_body_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.3, 1.0),
    #         "dynamic_friction_range": (0.3, 0.8),
    #         "restitution_range": (0.0, 0.5),
    #         "num_buckets": 64,
    #     },
    # )

    # UnitreeGo2 uses add_base_mass with base body only
    # randomize_rigid_body_mass_base = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-1.0, 3.0),
    #         "operation": "add",
    #         "recompute_inertia": True,
    #     },
    # )

    # randomize_rigid_body_mass_others = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=f"^(?!.*base).*"),
    #         "mass_distribution_params": (0.7, 1.3),
    #         "operation": "scale",
    #         "recompute_inertia": True,
    #     },
    # )

    # Skip: inertia updated via mass randomization by setting recompute_inertia=True
    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    # UnitreeGo2: base_com is set to None (removed)
    # 如果想要完全匹配 UnitreeGo2，应该设为 None 或删除此项
    # randomize_com_positions = None
    
    # 如果想保留 COM 随机化（增加训练多样性），保留以下配置
    # randomize_com_positions = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},  # z范围改为父类的(-0.01, 0.01)
    #     },
    # )

    # reset
    # UnitreeGo2 uses base_external_force_torque with base body
    # randomize_apply_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-10.0, 10.0),
    #     },
    # )

    # UnitreeGo2 uses position_range (1.0, 1.0) for reset_robot_joints
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        # func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # randomize_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.9, 1.1),
    #         "damping_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # UnitreeGo2 uses specific reset_base params
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
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

    # interval
    # UnitreeGo2: push_robot is set to None (removed)
    # 如果想要完全匹配 UnitreeGo2，应该设为 None 或删除此项
    # randomize_push_robot = None
    
    # 如果想保留周期性推机器人（增加训练鲁棒性），保留以下配置
    # randomize_push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )

@configclass
class Go2RoughActionsCfg:
    """为MDP配置动作值"""

    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=GO2_JOINT_NAMES, 
        scale=0.25,  # UnitreeGo2 uses 0.25 for all joints
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )
    
@configclass
class Go2RoughCommandsCfg:
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
            lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.7, 0.7), heading=(-math.pi, math.pi)
        ),
    )

    # 步态命令
    gait_command = mdp_nhb.QuadrupedGaitCommandCfg(
        resampling_time_range=(10, 10),
        ranges=mdp_nhb.QuadrupedGaitCommandCfg.Ranges(
            stance_rate=(0.5, 0.5),      # 支撑相占比50%
            rf_offset=(0.5, 0.5),        # 右前足相位偏移0.5 (Trot步态)
            lb_offset=(0.5, 0.5),        # 左后足相位偏移0.5 (Trot步态)
            rb_offset=(0.0, 0.0),        # 右后足相位偏移0.0 (与左前足同相)
            gait_frequency=(1.5, 1.5)  # 步态频率1.5Hz
        ),
    )     

@configclass
class Go2RoughTerminationsCfg:
    """复位触发条件"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # UnitreeGo2 uses base body only for base_contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
    )

@configclass
class Go2RoughCurriculumCfg:
    """课程"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)



@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # 开启图形渲染时才使用绘图窗口
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type:Go2DebugWindow = Go2DebugWindow
    scene: Go2RoughSceneCfg = Go2RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: Go2RoughObsCfg = Go2RoughObsCfg()
    actions: Go2RoughActionsCfg = Go2RoughActionsCfg()
    commands: Go2RoughCommandsCfg = Go2RoughCommandsCfg()
    # MDP settings
    rewards: Go2RoughRewards = Go2RoughRewards()
    terminations: Go2RoughTerminationsCfg = Go2RoughTerminationsCfg()
    events: Go2RoughEventCfg = Go2RoughEventCfg()
    curriculum: Go2RoughCurriculumCfg = Go2RoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # 仿真参数设置
        self.decimation = 4 # 算法50Hz
        self.sim.dt = 0.005 # 仿真200Hz
        self.episode_length_s = 20.0 # 1个episode20s

@configclass
class Go2FlatEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
