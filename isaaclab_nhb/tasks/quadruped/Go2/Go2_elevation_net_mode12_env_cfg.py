import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import math

import isaaclab_nhb
if not isaaclab_nhb.HEADLESS_FLAG:
    from isaaclab_nhb.envs.ui.manager_debug_rl_window import ManagerDebugRLEnvWindow
    from isaaclab_nhb.envs.manager_debug_rl_env import ManagerDebugRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor
from .Go2_env_cfg import Go2RoughEnvCfg, Go2RoughObsCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb
from .Go2_asset_cfg import GO2_CFG, GO2_JOINT_NAMES
from isaaclab_nhb.terrains.config.rough import ROUGH_ELEVATION_CFG, ROUGH_ELEVATION_CFG_GO2
from isaaclab.sensors import RayCaster
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

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
            # self.register_plot("foot rayCaster", ["foot height image"], data_type="image")
            self.register_plot("policy height map", ["policy height map"], data_type="image")
            self.register_plot("foot contact force x", ["FL", "FR", "RL", "RR"], data_type="plot")
            self.register_plot("foot contact force y", ["FL", "FR", "RL", "RR"], data_type="plot")
            self.register_plot("foot contact force z", ["FL", "FR", "RL", "RR"], data_type="plot")

        def _update_debug_data(self) -> dict:
            """更新调试数据"""
            data = {}

            # 足端高度图
            # left_raycast_sensor: RayCaster = self.env.scene.sensors["left_foot_height_scanner"]
            # right_raycast_sensor: RayCaster = self.env.scene.sensors["right_foot_height_scanner"]
            # left_raycaster_point = (left_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) -
            #                         left_raycast_sensor.data.ray_hits_w[:, :, 2])[0, :].reshape(27, 8)
            # right_raycaster_point = (right_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) -
            #                          right_raycast_sensor.data.ray_hits_w[:, :, 2])[0, :].reshape(27, 8)
            # # 创建中间的分割线
            # zeros_insert = torch.ones((left_raycaster_point.size(0), 1),
            #                           dtype=left_raycaster_point.dtype,
            #                           device=left_raycaster_point.device)
            # # 将左右传感器拼接成一张图片
            # image_to_show = torch.cat([left_raycaster_point, zeros_insert, right_raycaster_point], dim=1) * 10
            # data["foot rayCaster"] = image_to_show

            # policy高程图 - 从obs_buf中读取height_scan_policy数据
            # obs_buf["height_scan_policy"]的形状为 [num_envs, 1, 5, 25, 17]
            # 取第0个环境的第0个历史帧（最新帧）的5帧数据
            height_map_policy = self.env.obs_buf["height_scan_policy"][0, 0, :, :]  # [5, 25, 17]
            # 将5帧水平拼接成一张大图: [5, 25, 17] -> [25, 85]
            data["policy height map"] = height_map_policy

            # 足端触地力
            foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
            foot_contact_sensor: ContactSensor = self.env.scene.sensors[foot_sensor_cfg.name]
            
            # 获取四个足端的索引（假设按照body_names顺序）
            fl_idx = 6  # FL_foot
            fr_idx = 15  # FR_foot
            rl_idx = 26  # RL_foot
            rr_idx = 35  # RR_foot
            
            # x方向接触力
            fl_foot_frc_x = foot_contact_sensor.data.net_forces_w[0, fl_idx, 0]
            fr_foot_frc_x = foot_contact_sensor.data.net_forces_w[0, fr_idx, 0]
            rl_foot_frc_x = foot_contact_sensor.data.net_forces_w[0, rl_idx, 0]
            rr_foot_frc_x = foot_contact_sensor.data.net_forces_w[0, rr_idx, 0]
            data["foot contact force x"] = torch.stack([fl_foot_frc_x, fr_foot_frc_x, rl_foot_frc_x, rr_foot_frc_x])
            
            # y方向接触力
            fl_foot_frc_y = foot_contact_sensor.data.net_forces_w[0, fl_idx, 1]
            fr_foot_frc_y = foot_contact_sensor.data.net_forces_w[0, fr_idx, 1]
            rl_foot_frc_y = foot_contact_sensor.data.net_forces_w[0, rl_idx, 1]
            rr_foot_frc_y = foot_contact_sensor.data.net_forces_w[0, rr_idx, 1]
            data["foot contact force y"] = torch.stack([fl_foot_frc_y, fr_foot_frc_y, rl_foot_frc_y, rr_foot_frc_y])
            
            # z方向接触力
            fl_foot_frc_z = foot_contact_sensor.data.net_forces_w[0, fl_idx, 2]
            fr_foot_frc_z = foot_contact_sensor.data.net_forces_w[0, fr_idx, 2]
            rl_foot_frc_z = foot_contact_sensor.data.net_forces_w[0, rl_idx, 2]
            rr_foot_frc_z = foot_contact_sensor.data.net_forces_w[0, rr_idx, 2]
            data["foot contact force z"] = torch.stack([fl_foot_frc_z, fr_foot_frc_z, rl_foot_frc_z, rr_foot_frc_z])

            return data


@configclass
class Go2ElevationNetMode12ObsCfg(Go2RoughObsCfg):
    """Mode12观测配置: 本体观测 + 50帧高程图历史"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy观测组: 包含本体历史观测和高程图历史"""
        
        # base角速度 [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        # base重力向量 [3]
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        # 关节位置 [12]
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # 关节速度 [12]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 上一次的动作值 [12]
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HeightScanPolicyCfg(ObsGroup):
        """高程图历史配置: 从50帧历史中采样5帧"""

        # 高程图 [425] - 从50帧中隔10帧采样5帧
        height_scan = ObsTerm(
            func=mdp_nhb.height_scan_sampled,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"), 
                "interval_frames": 1,
                "total_frames": 5
            },
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic观测组: 包含完整的状态信息用于价值估计"""
        
        # pelvis角速度 [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # pelvis重力向量 [3]
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 关节位置 [12]
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES, preserve_order=True)}
        )
        # 关节速度 [12]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_NAMES, preserve_order=True)}
        )
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 上一次的动作值 [12]
        actions = ObsTerm(func=mdp.last_action)
        # base线速度 [3]
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightScanCriticCfg(ObsGroup):
        """高程图历史配置: 从50帧历史中采样5帧"""

        # 高程图 [425] - 从50帧中隔10帧采样5帧
        height_scan = ObsTerm(
            func=mdp_nhb.height_scan_sampled,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"), 
                "interval_frames": 1,
                "total_frames": 5
            },
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    height_scan_policy: HeightScanPolicyCfg = HeightScanPolicyCfg()
    critic: CriticCfg = CriticCfg()
    height_scan_critic: HeightScanCriticCfg = HeightScanCriticCfg()


@configclass
class Go2ElevationNetMode12RewardsCfg():
    """Mode12奖励配置"""

    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.5})

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
    # base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.34})
    base_height_exp = RewTerm(
        func=mdp_nhb.base_height_exp, 
        weight=1.0,  # 正权重
        params={
            "target_height": 0.34, 
            "std": 0.05,  # 调整敏感度
            "sensor_cfg": SceneEntityCfg("height_scanner")
        }
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    stand_still_vel = RewTerm(
        func=mdp_nhb.stand_still_without_cmd_vel_l1_l2, # 走复杂地形的时候不能使用普通standstill
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint.*")},
    )
    joint_deviation_all = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "command_name": "base_velocity",
            "threshold": 0.0,  # 腾空时间阈值（秒）
        },
    )
    # 足端穿越地面惩罚
    # feet_below_ground = RewTerm(
    #     func=mdp_nhb.feet_below_ground_penalty,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
    #         "threshold": 0.0,  # 地面高度阈值
    #     },
    # )
    # gait_rew = RewTerm(
    #     func=mdp_nhb.QuadrupedGaitReward,
    #     weight=4.0,
    #     params={
    #         "lf_sensor_cfg": SceneEntityCfg("contact_forces", body_names="FL_foot"),
    #         "rf_sensor_cfg": SceneEntityCfg("contact_forces", body_names="FR_foot"),
    #         "lb_sensor_cfg": SceneEntityCfg("contact_forces", body_names="RL_foot"),
    #         "rb_sensor_cfg": SceneEntityCfg("contact_forces", body_names="RR_foot"),

    #         "lf_asset_cfg": SceneEntityCfg("robot", body_names="FL_foot"),
    #         "rf_asset_cfg": SceneEntityCfg("robot", body_names="FR_foot"),
    #         "lb_asset_cfg": SceneEntityCfg("robot", body_names="RL_foot"),
    #         "rb_asset_cfg": SceneEntityCfg("robot", body_names="RR_foot"),

    #         "foot_height_tar": 0.1
    #     }
    # )

@configclass
class Go2ElevationNetMode12RoughEventCfg:
    """Configuration for events."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 256,
        },
    )
    # 关节摩擦力随机化
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 0.2),
            # "armature_distribution_params": (0.01, 0.01),
            "operation": "abs",
        },
    )
    # base link 的质量随机化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )
    # 随机每一个link的质量
    add_all_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.97, 1.03),
            "operation": "scale",
        },
    )

    # 惯量随机化
    randomize_base_inertias = EventTerm(
        func=mdp_nhb.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    # 质心随机化
    randomize_base_coms = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # 随机推动
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        min_step_count_between_reset= 100,
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.30, 0.30), "y": (-0.30, 0.30), "z": (-0.10, 0.10)}},
    )

    # 随机化kp kd
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # # 复位时会受到力的作用
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # 这个注释了就不会reset了，不能注
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # 复位时关节默认角度的随机化
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.85, 1.15),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class Go2ElevationNetMode12RoughActionsCfg:
    """为MDP配置动作值"""
    # 在asset设置了但在这里没有设置就是PD保持0位
    # 在asset没有设置就是不控制
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=GO2_JOINT_NAMES, 
        scale=0.25,  # UnitreeGo2 uses 0.25 for all joints
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)}
    )


@configclass
class Go2ElevationNetMode12CommandsCfg:
    """Command specifications for the MDP."""

    # 基于地形自适应的速度跟踪命令
    base_velocity = mdp_nhb.TerrainAdaptiveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # 速度采样范围（所有地形）
        ranges=mdp_nhb.TerrainAdaptiveVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), 
            lin_vel_y=(-0.5, 0.5), 
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi)
        ),
        # concentric_moats地形在最高难度时排除中间83%的低速区域
        # 例如: lin_vel_x在难度9时只采样[-1.2, -1.0]和[1.0, 1.2]
        moats_min_speed_threshold=0.8,
    )

@configclass
class Go2ElevationNetMode12CurriculumCfg:
    """课程"""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    # # 根据速度跟踪奖励逐步增加feet_contact_forces_xy的惩罚权重
    # feet_contact_forces_xy_weight = CurrTerm(
    #     func=mdp_nhb.reward_weight_tracking_levels,
    #     params={
    #         "reward_term_name": "feet_contact_forces_xy",
    #         "tracking_reward_name": "track_lin_vel_xy_exp",  # 参考线速度跟踪奖励
    #         "step": 0.01,  # 每次增加1%
    #         "reward_threshold": 0.65,  # 当速度跟踪达到权重的65%时开始增加
    #         "init_ratio": 0.0,  # 初始权重为0（从0开始逐步增加）
    #     },
    # )


@configclass
class Go2ElevationNetMode12TerminationsCfg:
    """复位触发条件"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # UnitreeGo2 uses base body only for base_contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*hip.*", ".*thigh.*"]), "threshold": 1.0},
    )


@configclass
class Go2ElevationNetMode12RoughEnvCfg(Go2RoughEnvCfg):
    # 开启图形渲染时才使用绘图窗口
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type: Go2DebugWindow = Go2DebugWindow

    observations: Go2ElevationNetMode12ObsCfg = Go2ElevationNetMode12ObsCfg()
    rewards: Go2ElevationNetMode12RewardsCfg = Go2ElevationNetMode12RewardsCfg()
    actions: Go2ElevationNetMode12RoughActionsCfg = Go2ElevationNetMode12RoughActionsCfg()
    commands: Go2ElevationNetMode12CommandsCfg = Go2ElevationNetMode12CommandsCfg()
    curriculum: Go2ElevationNetMode12CurriculumCfg = Go2ElevationNetMode12CurriculumCfg()
    terminations: Go2ElevationNetMode12TerminationsCfg = Go2ElevationNetMode12TerminationsCfg()
    events: Go2ElevationNetMode12RoughEventCfg = Go2ElevationNetMode12RoughEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # 只使用台阶和box地形
        self.scene.terrain.terrain_generator = ROUGH_ELEVATION_CFG_GO2
        # self.scene.terrain.terrain_generator = ROUGH_TERRAINS_PLANE_CFG
        self.scene.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.episode_length_s = 40.0 # 1个episode40s



@configclass
class Go2ElevationNetMode12FlatEnvCfg(Go2ElevationNetMode12RoughEnvCfg):
    """Mode12平坦地形环境配置"""
    
    def __post_init__(self):
        super().__post_init__()
        # 平坦地形相关调整
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None