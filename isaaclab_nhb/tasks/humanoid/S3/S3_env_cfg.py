import torch
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
import math
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RecorderManagerBaseCfg
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
from isaaclab_nhb import ISAACLAB_NHB_PATH

from .S3_asset_cfg import S3_12DOF_CFG, S3_12DOF_JOINT_LOWER_ORDER, S3_12DOF_FIX_BASE_CFG, S3_22DOF_JOINT_ORDER, S3_22DOF_CFG, S3_12DOF_JOINT_UPPER_ORDER, S3_22DOF_FIX_JOINTLIMIT_CFG, S3_22DOF_FIX_BASE_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG 
from dataclasses import MISSING
from isaaclab.assets import Articulation, RigidObject

if not isaaclab_nhb.HEADLESS_FLAG:
    class S3DebugWindow(ManagerDebugRLEnvWindow):
        """S3环境的调试窗口"""
        env: ManagerDebugRLEnv
        
        def __init__(
            self,
            env: ManagerDebugRLEnv,
            window_name="IsaacLab",
            debug_window_name="debug info",
            enable_obs_detail_plot=True,  # 是否绘制30维观测详情
        ):
            self.last_obs_predict = None
            self.enable_obs_detail_plot = enable_obs_detail_plot
            super().__init__(env, window_name, debug_window_name)

            # 观测值名称映射 (总共30维)
            self.obs_names = [
                # base角速度 [3]
                "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
                # base重力向量 [3]
                "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
                # 关节位置 [12]
                "left_hip_roll_pos", "left_hip_yaw_pos", "left_hip_pitch_pos",
                "left_knee_pos", "left_foot_pitch_pos", "left_foot_roll_pos",
                "right_hip_roll_pos", "right_hip_yaw_pos", "right_hip_pitch_pos",
                "right_knee_pos", "right_foot_pitch_pos", "right_foot_roll_pos",
                # 关节速度 [12]
                "left_hip_roll_vel", "left_hip_yaw_vel", "left_hip_pitch_vel",
                "left_knee_vel", "left_foot_pitch_vel", "left_foot_roll_vel",
                "right_hip_roll_vel", "right_hip_yaw_vel", "right_hip_pitch_vel",
                "right_knee_vel", "right_foot_pitch_vel", "right_foot_roll_vel",
            ]
        
        def _register_debug_plots(self):
            """注册需要的图表"""
            self.register_plot("vel_est_x", ["command", "estimate", "real"])
            self.register_plot("obs_predict_error", ["MSE", "MAE", "MAX"])
            self.register_plot("gait I", ["I_left", "I_right"])
            self.register_plot("sc command", ["sin_l", "cos_l", "sin_r", "cos_r"])
            
            # 如果启用30维观测详情绘制 - 为每个维度创建一个图表
            if self.enable_obs_detail_plot:
                for obs_name in self.obs_names:
                    self.register_plot(obs_name, ["real", "predict"], plot_height=120, collapsed=True)
        
        def _update_debug_data(self) -> dict:
            """更新调试数据"""
            data = {}
            
            # 速度估计
            if hasattr(self.env, "action_extra_info") and "est_vel" in self.env.action_extra_info:
                velocity_command = self.env.command_manager.get_command("base_velocity")
                velocity_estimate = self.env.action_extra_info["est_vel"].to(self.env.device)
                asset: RigidObject = self.env.scene["robot"]
                velocity_base = asset.data.root_lin_vel_b
                x_vel_cmd = velocity_command[0, 0]
                x_vel_est = velocity_estimate[0, 0]
                x_vel_base = velocity_base[0, 0]
                data["vel_est_x"] = torch.stack([x_vel_cmd, x_vel_est, x_vel_base])
            else:
                # 如果数据不可用，提供占位数据
                data["vel_est_x"] = torch.zeros(3, device=self.env.device)
            
            # 观测预测误差
            if hasattr(self.env, "action_extra_info") and "obs_predict" in self.env.action_extra_info:
                # 获取0号机器人的预测观测
                obs_predict = self.env.action_extra_info["obs_predict"].to(self.env.device)[0, :]
                
                # 初始化 last_obs_predict（也使用0号机器人）
                if self.last_obs_predict is None:
                    self.last_obs_predict = obs_predict.clone()
                
                # 获取0号机器人的真实观测
                obs_buffer = getattr(self.env.observation_manager, "_obs_buffer", None)
                
                if obs_buffer is None or (isinstance(obs_buffer, dict) and "obs_one_frame" not in obs_buffer):
                    policy_obs = torch.zeros_like(self.last_obs_predict, device=self.env.device)
                else:
                    policy_obs = obs_buffer["obs_one_frame"].to(self.env.device)[0, :30]
                
                # 计算误差（使用上一帧的预测和当前帧的真实值比较）
                obs_error_mse = torch.mean((policy_obs - self.last_obs_predict) ** 2)
                obs_error_mae = torch.mean(torch.abs(policy_obs - self.last_obs_predict))
                obs_error_max = torch.max(torch.abs(policy_obs - self.last_obs_predict))
                
                data["obs_predict_error"] = torch.stack([obs_error_mse, obs_error_mae, obs_error_max])
                
                # 更新 last_obs_predict 为当前的预测值
                self.last_obs_predict = obs_predict.clone()
                
                # 如果启用30维观测详情绘制 - 为每个维度填充数据
                if self.enable_obs_detail_plot:
                    policy_obs_flat = policy_obs.flatten()
                    predict_obs_flat = obs_predict.flatten()
                    
                    for i, obs_name in enumerate(self.obs_names):
                        data[obs_name] = torch.stack([
                            policy_obs_flat[i], 
                            predict_obs_flat[i]
                        ])
            else:
                # 如果数据不可用，提供占位数据
                data["obs_predict_error"] = torch.zeros(3, device=self.env.device)
                if self.enable_obs_detail_plot:
                    for obs_name in self.obs_names:
                        data[obs_name] = torch.zeros(2, device=self.env.device)
            
            # 相位指示器
            gait_rew = self.env.reward_manager._class_term_cfgs[0].func
            I_left = gait_rew.left_I[0]
            I_right = gait_rew.right_I[0]
            data["gait I"] = torch.stack([I_left, I_right])
            
            # 正余弦命令
            gait_command = self.env.command_manager.get_command("gait_command")
            data["sc command"] = torch.stack([
                gait_command[0, 3], gait_command[0, 4], 
                gait_command[0, 5], gait_command[0, 6]
            ])
            
            return data

@configclass
class S3RoughSceneCfg(InteractiveSceneCfg):
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
    robot: ArticulationCfg = S3_22DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
class S3RoughObsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # base角速度 [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.15, n_max=0.15))
        # base重力向量 [3]
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置 [12]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=S3_12DOF_JOINT_LOWER_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.03, n_max=0.03))
        # 关节速度 [12]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=S3_12DOF_JOINT_LOWER_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-1.8, n_max=1.8))
        # 上一次的动作值 [12]
        actions = ObsTerm(func=mdp.last_action)
        # 正余弦信息 [7]
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
        # 关节位置 [12]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=S3_12DOF_JOINT_LOWER_ORDER,
                                                            preserve_order=True)})
        # 关节速度 [12]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=S3_12DOF_JOINT_LOWER_ORDER,
                                                            preserve_order=True)})

        # 上一次的动作值
        actions = ObsTerm(func=mdp.last_action)
        # 正余弦信息
        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})
        # 高度采样
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
class S3RoughRewards:
    
    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=5.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})

    # 机体平衡奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # 运动平滑奖励
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.3) # 课程会使其逐步增大
    action_smooth_l2 = RewTerm(func=mdp_nhb.action_smooth_l2, weight=-0.3)

    # 安全奖励
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=S3_12DOF_JOINT_LOWER_ORDER)})
    joint_vel_soft_limits = RewTerm(func=mdp_nhb.joint_vel_soft_limits, weight=-1.0, params={"soft_ratio":0.9,"asset_cfg": SceneEntityCfg("robot", joint_names=S3_12DOF_JOINT_LOWER_ORDER)})
    joint_tor_soft_limits = RewTerm(func=mdp_nhb.joint_tor_soft_limits, weight=-1.0, params={"soft_ratio":0.9,"asset_cfg": SceneEntityCfg("robot", joint_names=S3_12DOF_JOINT_LOWER_ORDER)})
    joint_torque_limits = RewTerm(func=mdp.applied_torque_limits,weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=S3_12DOF_JOINT_LOWER_ORDER)})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # biped_distance_y_exp = RewTerm(func=mdp_nhb.biped_distance_y_l2, weight=-150, 
    #     params={"min_distance": 0.15, "max_distance": 0.6, 
    #             "min_distance_standing": 0.2, "max_distance_standing": 0.3,
    #             "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_roll_link")})

    # 步态奖励
    base_height = RewTerm(func=mdp.base_height_l2,weight=-1.0,params={"target_height": 0.8})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_roll_link"),
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
    # stand_still_joint_vel = RewTerm(
    #     func=mdp_nhb.stand_still_without_cmd_vel,
    #     weight=-1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #     },
    # )
    # stand_still_base_vel = RewTerm(
    #     func=mdp_nhb.stand_still_without_cmd_base_vel,
    #     weight=-1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #     },
    # )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_pitch_joint", ".*_foot_roll_joint"])},
    )
    gait_rew = RewTerm(
        func=mdp_nhb.BipedalGaitReward,
        weight=4.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_foot_roll_link"),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_foot_roll_link"),
            "left_asset_cfg": SceneEntityCfg("robot", body_names="left_foot_roll_link"),
            "right_asset_cfg": SceneEntityCfg("robot", body_names="right_foot_roll_link"),
            "foot_height_tar": 0.15,  # 足端高度目标值
        }
    )

@configclass
class S3RoughEventCfg:
    """Configuration for events."""
    # startup
    # # link的摩擦力、弹性系数随机化
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.05, 1.3),
    #         "dynamic_friction_range": (0.05, 1.3),
    #         "restitution_range": (0.0, 1.0),
    #         "num_buckets": 256,
    #     },
    # )
    # # 关节摩擦力随机化
    # joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "friction_distribution_params": (0.01, 0.1),
    #         "armature_distribution_params": (0.01, 0.1),
    #         "operation": "abs",
    #     },
    # )
    # # 随机化default pos
    # default_dof_pos = EventTerm(
    #     func=mdp_nhb.randomize_joint_default_pos_add,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "default_pose_add_limit": (-0.03, 0.03),
    #     },
    # )
    # base link 的质量随机化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 10.0),
            "operation": "add",
        },
    )
    # # 随机每一个link的质量
    # add_all_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )

    # # 惯量随机化
    # randomize_base_inertias = EventTerm(
    #     func=mdp_nhb.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #     },
    # )

    # 质心随机化
    randomize_base_coms = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # 随机推动
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-0.50, 0.50)}},
    )

    # 随机化kp kd
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # 随机化手臂关节目标位置
    random_joints_target_position = EventTerm(
        func=mdp_nhb.random_joints_target_position_by_scale,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=S3_12DOF_JOINT_UPPER_ORDER),
            "position_range": (0.0, 0.0), # 会被课程修改
        },
    )

    # 复位时会受到力的作用
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": (5.0, 5.0),
    #         "torque_range": (-5.0, 5.0),
    #     },
    # )

    # 这个注释了就不会reset了，不能注
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # 复位时关节默认角度的随机化，不能注
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class S3RoughActionsCfg:
    """为MDP配置动作值"""
    # 在asset设置了但在这里没有设置就是PD保持0位
    # 在asset没有设置就是不控制
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=S3_12DOF_JOINT_LOWER_ORDER, 
        # scale={
        #     ".*_hip_roll_.*": 0.5,
        #     ".*_hip_yaw_.*": 0.5,
        #     ".*_hip_pitch_.*": 0.5,
        #     ".*_knee_.*": 0.5,
        #     ".*_foot_pitch_.*": 1.0,
        #     ".*_foot_roll_.*": 1.0,
        # }, 
        scale=0.5,
        use_default_offset=True,
        preserve_order=True)


@configclass
class S3RoughCommandsCfg:
    """Command specifications for the MDP."""

    # 速度跟踪命令
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1, # 增大站立环境的采样概率
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.7, 0.7), heading=(-math.pi, math.pi)
            # lin_vel_x=(0.6, 0.65), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.3, 0.3), heading=(-math.pi, math.pi)
        ),
    )

    # 步态命令
    gait_command = mdp_nhb.BipedalGaitCommandCfg(
        resampling_time_range=(10, 10),
        ranges=mdp_nhb.BipedalGaitCommandCfg.Ranges(
            stance_rate=(0.5, 0.5), 
            bipedal_offset=(0.5, 0.5),
            gait_frequency=(1.25, 1.25) 
        ),
    )     

@configclass
class S3RoughTerminationsCfg:
    """复位触发条件"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces", 
            body_names=[
                "base_link",
                ".*_shoulder_.*",
                ".*_elbow_.*",
            ]
        ),
                "threshold": 1.0},
    )

@configclass
class S3RoughCurriculumCfg:
    """课程"""
    # 地形课程
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # # 奖励权重课程
    # feet_distance_rew_weight_ratio = CurrTerm(
    #     func=mdp_nhb.reward_weight_tracking_levels,
    #     params={
    #         "reward_term_name": "biped_distance_y_exp",
    #         "step": 0.01,
    #         "reward_threshold": 0.6,
    #         "init_ratio": 0.1
    #     },
    # )
    action_rate_l2_rew_weight_ratio = CurrTerm(
        func=mdp_nhb.reward_weight_tracking_levels,
        params={
            "reward_term_name": "action_rate_l2",
            "step": 0.01,
            "reward_threshold": 0.6,
            "init_ratio": 0.01
        },
    )
    action_smooth_l2_rew_weight_ratio = CurrTerm(
        func=mdp_nhb.reward_weight_tracking_levels,
        params={
            "reward_term_name": "action_smooth_l2",
            "step": 0.01,
            "reward_threshold": 0.6,
            "init_ratio": 0.01
        },
    )
    # dof_smooth_l2_rew_weight_ratio = CurrTerm(
    #     func=mdp_nhb.reward_weight_tracking_levels,
    #     params={
    #         "reward_term_name": "dof_smooth_l2",
    #     },
    # )
    # flat_orientation_l1_rew_weight_ratio = CurrTerm(
    #     func=mdp_nhb.reward_weight_episodes_levels,
    #     params={
    #         "reward_term_name": "flat_orientation_l1",
    #         "init_ratio": 0.5,
    #     },
    # )
    # flat_orientation_l2_rew_weight_ratio = CurrTerm(
    #     func=mdp_nhb.reward_weight_episodes_levels,
    #     params={
    #         "reward_term_name": "flat_orientation_l2",
    #         "init_ratio": 0.5,
    #     },
    # )


    # 随机化程度课程
    # random_push_x_vel_levels = CurrTerm(
    #     func=mdp_nhb.random_push_x_vel_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "step": 0.02,
    #         "max_vel": 2.0,
    #     }
    # )
    # random_push_y_vel_levels = CurrTerm(
    #     func=mdp_nhb.random_push_y_vel_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "step": 0.02,
    #         "max_vel": 2.0,
    #     }
    # )
    # random_push_z_vel_levels = CurrTerm(
    #     func=mdp_nhb.random_push_z_vel_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "step": 0.01,
    #         "max_vel": 0.2,
    #     }
    # )

    # 训练下半身时随机化上半身关节位置
    random_joints_target_position_levels = CurrTerm(
        func=mdp_nhb.random_joints_target_position_level,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "reward_threshold": 0.65,
            "step": 0.015,
            "high_clip": 0.60,  # soft_limit是0.6
        }
    )
    # lin_vel_cmd_std_levels_exp = CurrTerm(
    #     func=mdp_nhb.lin_vel_cmd_std_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "step": 0.1,
    #         "min_std": 0.1
    #     }
    # )
    # lin_vel_cmd_std_levels_exp_y = CurrTerm(
    #     func=mdp_nhb.lin_vel_cmd_std_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_y_exp",
    #         "step": 0.01,
    #         "min_std": 0.05
    #     }
    # )

@configclass
class S3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type:S3DebugWindow = S3DebugWindow

    scene: S3RoughSceneCfg = S3RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: S3RoughObsCfg = S3RoughObsCfg()
    actions: S3RoughActionsCfg = S3RoughActionsCfg()
    commands: S3RoughCommandsCfg = S3RoughCommandsCfg()
    # MDP settings
    rewards: S3RoughRewards = S3RoughRewards()
    terminations: S3RoughTerminationsCfg = S3RoughTerminationsCfg()
    events: S3RoughEventCfg = S3RoughEventCfg()
    curriculum: S3RoughCurriculumCfg = S3RoughCurriculumCfg()


    def __post_init__(self):
        super().__post_init__()
        # 仿真参数设置
        self.decimation = 4 # 算法50Hz
        self.sim.dt = 0.005 # 仿真200Hz
        self.episode_length_s = 20.0 # 1个episode20s




@configclass
class S3FlatEnvCfg(S3RoughEnvCfg):
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


@configclass
class S3FlatEnvCfg_Check_USD(S3FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.robot = S3_12DOF_FIX_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
        self.terminations.base_contact = None
        self.terminations.time_out = None



