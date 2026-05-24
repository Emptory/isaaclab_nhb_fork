import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
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
from isaaclab.sensors import ContactSensor

import isaaclab_nhb 
if not isaaclab_nhb.HEADLESS_FLAG:
    from isaaclab_nhb.envs.ui.manager_debug_rl_window import ManagerDebugRLEnvWindow
    from isaaclab_nhb.envs.manager_debug_rl_env import ManagerDebugRLEnv


import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb
from isaaclab_nhb.terrains.config.rough import ROUGH_TERRAINS_SIMPLE_CFG
from isaaclab_nhb.tasks.mdp_nhb.curriculums import lin_vel_cmd_levels, ang_vel_cmd_levels
# from isaaclab_nhb.tasks.mdp_nhb.commands_cfg import UniformLevelVelocityCommandCfg

# from .G1_asset_cfg import G1_29DOF_CFG, G1_29DOF_CFG_FIXED_BASE, G1_29DOF_JOINT_ORDER, G1_26DOF_CFG, G1_26DOF_JOINT_ORDER, G1_12DOF_CFG, G1_29DOF_JOINT_ORDER, G1_12DOF_CFG_FIXED_BASE
from .G1_asset_cfg import G1_29DOF_CFG, G1_29DOF_JOINT_ORDER, G1_29DOF_ACTION_SCALE
from .G1_asset_cfg import G1_12DOF_CFG, G1_12DOF_JOINT_ORDER, G1_12DOF_ACTION_SCALE
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera 

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from dataclasses import MISSING


if not isaaclab_nhb.HEADLESS_FLAG:
    class G1DebugWindow(ManagerDebugRLEnvWindow):
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
            self.register_plot("foot rayCaster", ["foot height image"], plot_type="image")
            self.register_plot("foot contact force", ["l", "r"])
            self.register_plot("gait I", ["I_left", "I_right"])
            self.register_plot("sc command", ["sin_l", "cos_l", "sin_r", "cos_r"])

        def _update_debug_data(self) -> dict:
            """更新调试数据"""
            data = {}
            
            # 足端高度图
            left_raycast_sensor: RayCaster = self.env.scene.sensors["left_foot_height_scanner"]
            right_raycast_sensor: RayCaster = self.env.scene.sensors["right_foot_height_scanner"]
            left_raycaster_point = (left_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) - left_raycast_sensor.data.ray_hits_w[:,:,2])[0, :].reshape(21,6)
            right_raycaster_point = (right_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) - right_raycast_sensor.data.ray_hits_w[:,:,2])[0, :].reshape(21,6)
            # 创建中间的分割线
            zeros_insert = torch.ones((left_raycaster_point.size(0), 1), 
                          dtype=left_raycaster_point.dtype,
                          device=left_raycaster_point.device)
            # 将左右传感器拼接成一张图片
            image_to_show = torch.cat([left_raycaster_point, zeros_insert, right_raycaster_point], dim=1) * 10
            data["foot rayCaster"] = image_to_show

            # 足端触地力
            l_foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link"])
            r_foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["right_ankle_roll_link"])
            l_foot_contact_sensor: ContactSensor = self.env.scene.sensors[l_foot_sensor_cfg.name]
            r_foot_contact_sensor: ContactSensor = self.env.scene.sensors[r_foot_sensor_cfg.name]
            l_foot_frc_z = l_foot_contact_sensor.data.net_forces_w[0, 7, 2]
            r_foot_frc_z = r_foot_contact_sensor.data.net_forces_w[0, 14, 2]
            data["foot contact force"] = torch.stack([l_foot_frc_z, r_foot_frc_z])

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
class G1RoughSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_SIMPLE_CFG,
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = G1_12DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(  # 总的高程图
        prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.8], ordering="yx"),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.35, 0.95], ordering="yx"),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    # left_foot_height_scanner = RayCasterCfg(  # 左足高程图，用于触地面积奖励
    #     prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, 0.13)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.20, 0.07], ordering="yx"),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    # right_foot_height_scanner = RayCasterCfg(  # 右足高程图，用于触地面积奖励
    #     prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.03)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.20, 0.07], ordering="yx"),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )


    # left_foot_height_scanner_lidar = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
    #     prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.03)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    # left_foot_height_scanner_lidar_sec = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
    #     prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.01)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # right_foot_height_scanner_lidar = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
    #     prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.03)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    # right_foot_height_scanner_lidar_sec = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
    #     prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.01)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    
    contact_forces = ContactSensorCfg(  # 接触力传感器
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class G1RoughObsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # base角速度 [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        # base重力向量 [3]
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.03, n_max=0.03))
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置 [12]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=G1_12DOF_JOINT_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度 [12]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=G1_12DOF_JOINT_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-1.5, n_max=1.5))
        # 上一次的动作值 [12]
        actions = ObsTerm(func=mdp.last_action)
        # 步态信息 [7]
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
                                                            joint_names=G1_12DOF_JOINT_ORDER,
                                                            preserve_order=True)})
        # 关节速度 [12]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                    params={
                        "asset_cfg": SceneEntityCfg("robot", 
                                                    joint_names=G1_12DOF_JOINT_ORDER,
                                                    preserve_order=True)})
        # 上一次的动作值
        actions = ObsTerm(func=mdp.last_action)
        # 正余弦信息
        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})
        # 高度采样
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"),"offset": 0.0},
            clip=(-10.0, 10.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测值配置组
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class G1RoughRewardsCfg:

    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp,weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, 
        params={"command_name": "base_velocity", "std": 0.5})

    # 机体平衡奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-5.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)

    # 运动平滑奖励
    # action_smooth_tripple_l2 = RewTerm(func=mdp_nhb.action_smooth_tripple_l2,weight=-5e-4,params={"c_a":0.01, "c_da":2.5, "c_dda":1})
    dof_smooth_l2 = RewTerm(func=mdp_nhb.dof_smooth_l2,weight=-2e-6,params={"c_dof_v":0.02, "c_dof_a":0.025, "c_tor":1})
    body_smooth_l2 = RewTerm(func=mdp_nhb.body_smooth_l2,weight=-1e-5,
        params={
            "c_body_a":10, 
            "c_feet_a":1, 
            "base_asset_cfg":SceneEntityCfg("robot", body_names="pelvis"),
            "feet_asset_cfg":SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        }
    )

    # 安全奖励
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-7e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])}
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    joint_vel_soft_limits = RewTerm(func=mdp_nhb.joint_vel_soft_limits, weight=-0.2, params={"soft_ratio":0.95})
    joint_tor_soft_limits = RewTerm(func=mdp_nhb.joint_tor_soft_limits, weight=-0.2, params={"soft_ratio":0.95})
    # feet_contact_force = RewTerm(func=mdp.contact_forces, weight=-0.01, 
    #     params={"threshold": 480, "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")})
    biped_distance_y_exp = RewTerm(func=mdp_nhb.biped_distance_y_l2, weight=-150, 
        params={"min_distance": 0.1, "max_distance": 0.35, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 步态奖励
    base_height = RewTerm(func=mdp.base_height_l2,weight=-1.0,params={"target_height": 0.75}) 
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    stand_still = RewTerm(
        func=mdp_nhb.stand_still_without_cmd,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot",joint_names=[".*_shoulder_roll_.*", ".*_shoulder_yaw_.*",".*_wrist_.*"])},
    # )
    # joint_deviation_arms = RewTerm( # 限制手臂所有关节
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot",joint_names=[".*shoulder.*", ".*elbow.*",".*wrist.*"])},
    # )
    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    # )
    gait_rew = RewTerm(
        func=mdp_nhb.BipedalGaitReward,
        weight=3.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            "right_sensor_cfg":SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
            "left_asset_cfg":SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
            "right_asset_cfg":SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
            "foot_height_tar": 0.10,  # 足端高度目标值
        }
    )
    gait_rew_ensure = RewTerm(
        func=mdp_nhb.BipedalGaitEnsureReward,
        weight=1.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            "right_sensor_cfg":SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
        }
    )
    # human_shoulder_trajectory_rew = RewTerm(
    #     func=mdp_nhb.human_shoulder_trajectory_l2,
    #     weight=0.5,
    #     params={
    #         "left_shoulder_cfg":SceneEntityCfg("robot",joint_names=["left_shoulder_pitch_joint"]),
    #         "right_shoulder_cfg":SceneEntityCfg("robot",joint_names=["right_shoulder_pitch_joint"]),
    #         "left_elbow_cfg":SceneEntityCfg("robot",joint_names=["left_elbow_joint"]),
    #         "right_elbow_cfg":SceneEntityCfg("robot",joint_names=["right_elbow_joint"]),
    #         "shoulder_pitch_center": 0.131,
    #         "elbow_center": 0.9975,
    #         "std": 0.3
    #     }
    # )
    

@configclass
class G1RoughEventCfg:
    """Configuration for events."""
    # startup
    # link的摩擦力、弹性系数随机化
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.1, 1.0),
    #         "dynamic_friction_range": (0.1, 0.8),
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
    #         "friction_distribution_params": (0.01, 0.2),
    #         # "armature_distribution_params": (0.01, 0.01),
    #         "operation": "abs",
    #     },
    # )
    # # base link 的质量随机化
    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "mass_distribution_params": (-0.5, 2.0),
    #         "operation": "add",
    #     },
    # )
    # # 随机每一个link的质量
    # add_all_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #     },
    # )

    # # 惯量随机化
    # randomize_base_inertias = EventTerm(
    #     func=mdp_nhb.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )

    # # 质心随机化
    # randomize_base_coms = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    # # 随机推动
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     min_step_count_between_reset= 100,
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (-0.10, 0.10)}},
    # )

    # # 随机化kp kd
    # randomize_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # # 复位时会受到力的作用
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # 这个注释了就不会reset了，不能注
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
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

    # 复位时关节默认角度的随机化
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class G1RoughActionsCfg:
    """为MDP配置动作值"""
    # 在asset设置了但在这里没有设置就是PD保持0位
    # 在asset没有设置就是不控制
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=G1_12DOF_JOINT_ORDER, 
        scale=G1_12DOF_ACTION_SCALE, 
        use_default_offset=True,
        preserve_order=True,
        clip={
            ".*": (-50.0, 50.0),
        }
)


@configclass
class G1RoughCommandsCfg:
    """Command specifications for the MDP."""

    # 速度跟踪命令
    base_velocity = mdp_nhb.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.8,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
        limit_ranges= mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        )
    )

    # 步态命令
    gait_command = mdp_nhb.BipedalGaitCommandCfg(
        resampling_time_range=(10, 10),
        ranges=mdp_nhb.BipedalGaitCommandCfg.Ranges(
            stance_rate=(0.5, 0.5),    # 建议0.5-1.0
            bipedal_offset=(0.5, 0.5), # 建议0-0.5
            gait_frequency=(1.25, 1.25)  # 建议0-2Hz
        ),
    )     

@configclass
class G1RoughTerminationsCfg:
    """复位触发条件"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces", 
            body_names=[
                "pelvis",
                ".*knee_link",
            ]
        ), 
        "threshold": 1.0},
    )

@configclass
class G1RoughCurriculumCfg:
    """课程"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # 奖励权重课程
    feet_distance_rew_weight_ratio = CurrTerm(
        func=mdp_nhb.reward_weight_tracking_levels,
        params={
            "reward_term_name": "biped_distance_y_exp",
        },
    )
    action_smooth_tripple_l2_rew_weight_ratio = CurrTerm(
        func=mdp_nhb.reward_weight_tracking_levels,
        params={
            "reward_term_name": "action_smooth_tripple_l2",
        },
    )
    dof_smooth_l2_rew_weight_ratio = CurrTerm(
        func=mdp_nhb.reward_weight_tracking_levels,
        params={
            "reward_term_name": "dof_smooth_l2",
        },
    )

    # 随机化程度课程
    # random_push_x_vel_levels = CurrTerm(
    #     func=mdp_nhb.random_push_x_vel_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "step": 0.02,
    #         "max_vel": 1.0,
    #     }
    # )
    # random_push_y_vel_levels = CurrTerm(
    #     func=mdp_nhb.random_push_y_vel_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "step": 0.02,
    #         "max_vel": 1.0,
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
    # lin_vel_cmd_levels = CurrTerm(
    #     func=lin_vel_cmd_levels,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "command_term_name": "base_velocity",
    #         "step": 0.02
    #     }
    # )
    # ang_vel_cmd_levels = CurrTerm(
    #     func=ang_vel_cmd_levels,
    #     params={
    #         "reward_term_name": "track_ang_vel_z_exp",
    #         "command_term_name": "base_velocity",
    #         "step": 0.02
    #     }
    # )
    



@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # 开启图形渲染时才使用绘图窗口
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type:G1DebugWindow = G1DebugWindow
    scene: G1RoughSceneCfg = G1RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: G1RoughObsCfg = G1RoughObsCfg()
    actions: G1RoughActionsCfg = G1RoughActionsCfg()
    commands: G1RoughCommandsCfg = G1RoughCommandsCfg()
    # MDP settings
    rewards: G1RoughRewardsCfg = G1RoughRewardsCfg()
    terminations: G1RoughTerminationsCfg = G1RoughTerminationsCfg()
    events: G1RoughEventCfg = G1RoughEventCfg()
    curriculum: G1RoughCurriculumCfg = G1RoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # 仿真参数设置
        self.decimation = 4 # 算法50Hz
        self.sim.dt = 0.005 # 仿真200Hz
        self.episode_length_s = 20.0 # 1个episode20s


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    # 开启图形渲染时才使用绘图窗口
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type:G1DebugWindow = G1DebugWindow
    rewards: G1RoughRewardsCfg = G1RoughRewardsCfg()
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

