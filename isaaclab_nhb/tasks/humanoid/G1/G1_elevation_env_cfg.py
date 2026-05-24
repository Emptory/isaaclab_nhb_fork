import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import math
import isaaclab.sim as sim_utils
import isaaclab_nhb
if not isaaclab_nhb.HEADLESS_FLAG:
    from isaaclab_nhb.envs.ui.manager_debug_rl_window import ManagerDebugRLEnvWindow
    from isaaclab_nhb.envs.manager_debug_rl_env import ManagerDebugRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor
from isaaclab.scene import InteractiveSceneCfg
from .G1_env_cfg import G1RoughEnvCfg, G1RoughObsCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb
from .G1_asset_cfg import G1_29DOF_CFG, G1_29DOF_JOINT_ORDER, G1_29DOF_ACTION_SCALE
from isaaclab_nhb.terrains.config.rough import ROUGH_TERRAINS_STAIRS_CFG, ROUGH_ELEVATION_CFG, ROUGH_ELEVATION_RICH_CFG, ROUGH_ELEVATION_RICH_FOR_TEST_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.sensors import RayCaster
from .G1_AMP_env_cfg import G1AmpPureRewardsCfg, G1AmpRoughCurriculumCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from .G1_env_cfg import G1RoughSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

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
            self.register_plot("foot rayCaster", ["foot height image"], data_type="image")
            self.register_plot("policy height map", ["policy height map"], data_type="image")
            self.register_plot("foot contact force x", ["l", "r"], data_type="plot")
            self.register_plot("foot contact force y", ["l", "r"], data_type="plot")
            self.register_plot("foot contact force z", ["l", "r"], data_type="plot")

        def _update_debug_data(self) -> dict:
            """更新调试数据"""
            data = {}

            # 足端高度图
            left_raycast_sensor: RayCaster = self.env.scene.sensors["left_foot_height_scanner"]
            right_raycast_sensor: RayCaster = self.env.scene.sensors["right_foot_height_scanner"]
            left_raycaster_point = (left_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) -
                                    left_raycast_sensor.data.ray_hits_w[:, :, 2])[0, :].reshape(21, 8)
            right_raycaster_point = (right_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) -
                                     right_raycast_sensor.data.ray_hits_w[:, :, 2])[0, :].reshape(21, 8)
            # 创建中间的分割线
            zeros_insert = torch.ones((left_raycaster_point.size(0), 1),
                                      dtype=left_raycaster_point.dtype,
                                      device=left_raycaster_point.device)
            # 将左右传感器拼接成一张图片
            image_to_show = torch.cat([left_raycaster_point, zeros_insert, right_raycaster_point], dim=1) * 10
            data["foot rayCaster"] = image_to_show

            # policy高程图 - 从obs_buf中读取height_scan_policy数据
            # obs_buf["height_scan_policy"]的形状为 [num_envs, 1, 5, 25, 17]
            # 取第0个环境的第0个历史帧（最新帧）的5帧数据
            height_map_policy = self.env.obs_buf["height_scan_policy"][0, 0, :, :]  # [5, 25, 17]
            # 将5帧水平拼接成一张大图: [5, 25, 17] -> [25, 85]
            data["policy height map"] = height_map_policy

            # 足端触地力
            l_foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link"])
            r_foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["right_ankle_roll_link"])
            l_foot_contact_sensor: ContactSensor = self.env.scene.sensors[l_foot_sensor_cfg.name]
            r_foot_contact_sensor: ContactSensor = self.env.scene.sensors[r_foot_sensor_cfg.name]
            
            # x方向接触力
            l_foot_frc_x = l_foot_contact_sensor.data.net_forces_w[0, 7, 0]
            r_foot_frc_x = r_foot_contact_sensor.data.net_forces_w[0, 14, 0]
            data["foot contact force x"] = torch.stack([l_foot_frc_x, r_foot_frc_x])
            
            # y方向接触力
            l_foot_frc_y = l_foot_contact_sensor.data.net_forces_w[0, 7, 1]
            r_foot_frc_y = r_foot_contact_sensor.data.net_forces_w[0, 14, 1]
            data["foot contact force y"] = torch.stack([l_foot_frc_y, r_foot_frc_y])
            
            # z方向接触力
            l_foot_frc_z = l_foot_contact_sensor.data.net_forces_w[0, 7, 2]
            r_foot_frc_z = r_foot_contact_sensor.data.net_forces_w[0, 14, 2]
            data["foot contact force z"] = torch.stack([l_foot_frc_z, r_foot_frc_z])

            return data


@configclass
class G1ElevHistSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_ELEVATION_RICH_CFG,
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
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(  # 总的高程图
        prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.8], ordering="yx"),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[1.35, 0.95], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # TODO: 研究这个分辨率是否足够计算边缘
    left_foot_height_scanner = RayCasterCfg(  # 左足高程图，用于触地面积奖励
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, 0.13)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.20, 0.07], ordering="yx"),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    right_foot_height_scanner = RayCasterCfg(  # 右足高程图，用于触地面积奖励
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.03)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.20, 0.07], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


    left_foot_height_scanner_lidar = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.03)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    left_foot_height_scanner_lidar_sec = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.01)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    right_foot_height_scanner_lidar = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.03)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    right_foot_height_scanner_lidar_sec = RayCasterCfg(  # 左足雷达型射线扫描，用于碰撞惩罚
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, -0.01)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=[-0.0, 0.0], horizontal_fov_range=[-180, 180], horizontal_res=5.0), 
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
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
class G1ElevHistObsCfg(G1RoughObsCfg):
    """Mode13观测配置: 本体观测 + 50帧高程图历史"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy观测组: 包含本体历史观测和高程图历史"""
        
        # === 本体观测部分(需要历史) ===
        # pelvis角速度 [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        # pelvis重力向量 [3]
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        # 关节位置 [29]
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # 关节速度 [29]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 上一次的动作值 [29]
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
            noise=Unoise(n_min=-0.015, n_max=0.015),
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
        # 关节位置 [29]
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)}
        )
        # 关节速度 [29]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_ORDER, preserve_order=True)}
        )
        # pelvis线速度 [3]
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 上一次的动作值 [29]
        actions = ObsTerm(func=mdp.last_action)


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
class G1ElevHistRewardsCfg():
    """Mode13奖励配置"""

    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=5.0, 
        params={"command_name": "base_velocity", "std": 0.5})

    # 机体平衡奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    adaptive_orientation_l2 = RewTerm(
        func=mdp_nhb.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=-6.0
    )

    # 运动平滑奖励
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) 
    action_smooth_l2 = RewTerm(func=mdp_nhb.action_smooth_l2, weight=-0.01)

    # 安全奖励
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)
    joint_vel_soft_limits = RewTerm(func=mdp_nhb.joint_vel_soft_limits, weight=-1.0, params={"soft_ratio": 0.9})
    joint_tor_soft_limits = RewTerm(func=mdp_nhb.joint_tor_soft_limits, weight=-1.0, params={"soft_ratio": 0.9})
    joint_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.2)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-250.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*shoulder_.*", ".*elbow_.*", ".*wrist_.*", ".*hand", ".*hip_yaw.*", ".*torso_.*"]), "threshold": 1.0},
    )

    base_height_exp = RewTerm(
        func=mdp_nhb.base_height_exp, 
        weight=1.0,  # 正权重
        params={
            "target_height": 0.75, 
            "std": 0.05,  # 调整敏感度
            "sensor_cfg": SceneEntityCfg("height_scanner")
        }
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # 关节偏差惩罚-需要大幅度运动的关节少惩罚,反之多惩罚
    joint_deviation_movement_joints = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_pitch.*", ".*knee.*", ".*ankle_pitch.*", ".*shoulder_pitch.*", ".*elbow.*", ".*wrist.*"])}
    )

    joint_deviation_static_joints = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*", ".*hip_yaw.*", ".*ankle_roll.*", "waist.*", ".*shoulder_roll.*", ".*shoulder_yaw.*", ".*wrist.*"])}
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=7.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 1.0,
        },
    )

    # 足端碰撞检测奖励（基于射线检测楼梯和障碍物）
    feet_collision_penalty = RewTerm(
        func=mdp_nhb.FeetCollisionPenalty,
        weight=-500.0,
        params={
            "raycaster_sensor_cfgs": [
                SceneEntityCfg("left_foot_height_scanner_lidar"),
                SceneEntityCfg("left_foot_height_scanner_lidar_sec"),
                SceneEntityCfg("right_foot_height_scanner_lidar"),
                SceneEntityCfg("right_foot_height_scanner_lidar_sec"),
            ],
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "d_safe": 0.15,
            # "d_safe": 0.20, # 测试用
            "slope_threshold": 1.0, # 对应78.5度
            "cone_angle": 30.0,
            "vel_buffer_size": 3
        },
    )
    
    # 地形边缘惩罚（基于平面拟合检测台阶、沟壑等非平面地形）
    terrain_edge_penalty = RewTerm(
        func=mdp_nhb.TerrainEdgePenalty,
        weight=2.0,
        params={
            "raycaster_sensor_cfgs": [
                SceneEntityCfg("left_foot_height_scanner"),
                SceneEntityCfg("right_foot_height_scanner")
            ],
            "d_sens": 0.10,
            "dist_offset": 0.035,
            "edge_threshold": 0.03,
        }
    )

@configclass
class G1ElevHistRoughEventCfg:
    """Configuration for events."""
    # startup
    # link的摩擦力、弹性系数随机化
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
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
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
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

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
class G1ElevHistRoughActionsCfg:
    """为MDP配置动作值"""
    # 在asset设置了但在这里没有设置就是PD保持0位
    # 在asset没有设置就是不控制
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_JOINT_ORDER,
        scale=G1_29DOF_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={
            ".*": (-50.0, 50.0),
        }
    )

@configclass
class G1ElevHistCommandsCfg:
    """Command specifications for the MDP."""

    # 速度跟踪命令
    base_velocity = mdp_nhb.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class G1ElevHistCurriculumCfg:
    """课程"""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    # 根据速度跟踪奖励逐步增加feet_contact_forces_xy的惩罚权重
    feet_collision_penalty_weight = CurrTerm(
        func=mdp_nhb.reward_weight_tracking_levels,
        params={
            "reward_term_name": "feet_collision_penalty",
            "tracking_reward_name": "track_lin_vel_xy_exp",  # 参考线速度跟踪奖励
            "step": 0.02,  # 每次增加1%
            "reward_threshold": 0.65,  # 当速度跟踪达到权重的65%时开始增加
            "init_ratio": 0.0,  # 初始权重为0（从0开始逐步增加）
        },
    )
    terrain_edge_penalty_weight = CurrTerm(
        func=mdp_nhb.reward_weight_tracking_levels,
        params={
            "reward_term_name": "terrain_edge_penalty",
            "tracking_reward_name": "track_lin_vel_xy_exp",  # 参考线速度跟踪奖励
            "step": 0.02,  # 每次增加1%
            "reward_threshold": 0.65,  # 当速度跟踪达到权重的65%时开始增加
            "init_ratio": 0.0,  # 初始权重为0（从0开始逐步增加）
        },
    )

    # action平滑奖励课程
    # action_rate_l2_weight = CurrTerm(
    #     func=mdp_nhb.reward_weight_tracking_levels,
    #     params={
    #         "reward_term_name": "action_rate_l2",
    #         "tracking_reward_name": "track_lin_vel_xy_exp",  # 参考线速度跟踪奖励
    #         "step": 0.02,  # 每次增加1%
    #         "reward_threshold": 0.65,  # 当速度跟踪达到权重的65%时开始增加
    #         "init_ratio": 0.0,  # 初始权重为0（从0开始逐步增加）
    #     },
    # )

    # action_smooth_l2_weight = CurrTerm(
    #     func=mdp_nhb.reward_weight_tracking_levels,
    #     params={
    #         "reward_term_name": "action_smooth_l2",
    #         "tracking_reward_name": "track_lin_vel_xy_exp",  # 参考线速度跟踪奖励
    #         "step": 0.02,  # 每次增加1%
    #         "reward_threshold": 0.65,  # 当速度跟踪达到权重的65%时开始增加
    #         "init_ratio": 0.0,  # 初始权重为0（从0开始逐步增加）
    #     },
    # )


@configclass
class G1ElevHistTerminationsCfg:
    """复位触发条件"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[
                "pelvis.*",
                ".*torso_.*",
                ".*head.*",
                # ".*hip_yaw.*",
                # ".*shoulder.*",
                # ".*elbow.*"
            ]
        ),
            "threshold": 1.0},
    )

@configclass
class G1ElevHistRoughEnvCfg(G1RoughEnvCfg):
    # 开启图形渲染时才使用绘图窗口
    # TODO: 限制raycaster按照50Hz更新
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type: G1DebugWindow = G1DebugWindow

    scene: G1ElevHistSceneCfg = G1ElevHistSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1ElevHistObsCfg = G1ElevHistObsCfg()
    rewards: G1ElevHistRewardsCfg = G1ElevHistRewardsCfg()
    actions: G1ElevHistRoughActionsCfg = G1ElevHistRoughActionsCfg()
    commands: G1ElevHistCommandsCfg = G1ElevHistCommandsCfg()
    curriculum: G1ElevHistCurriculumCfg = G1ElevHistCurriculumCfg()
    terminations: G1ElevHistTerminationsCfg = G1ElevHistTerminationsCfg()
    events: G1ElevHistRoughEventCfg = G1ElevHistRoughEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # self.scene.terrain.terrain_generator = ROUGH_ELEVATION_CFG

        self.episode_length_s = 40.0 # 1个episode40s

############################################# 加入了AMP ###############################################

@configclass
class G1ElevHistAMPObsCfg(G1ElevHistObsCfg):
    """Mode13观测配置: 本体观测 + 50帧高程图历史"""
    @configclass
    class AMPObsCfg(ObsGroup):
        """Observations for AMP group."""
        joint_pos = ObsTerm(func=mdp.joint_pos,params={"asset_cfg": SceneEntityCfg("robot",joint_names=G1_29DOF_JOINT_ORDER,preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel,params={"asset_cfg": SceneEntityCfg("robot",joint_names=G1_29DOF_JOINT_ORDER,preserve_order=True)})
        end_bodies_pos = ObsTerm(func=mdp_nhb.bodies_pos_order_r,params={
                                            "bodies_order": [
                                                "left_wrist_yaw_link","right_wrist_yaw_link",
                                                "left_ankle_roll_link","right_ankle_roll_link",
                                                ]})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    amp: AMPObsCfg = AMPObsCfg()

@configclass
class G1ElevHistAMPRoughEnvCfg(G1ElevHistRoughEnvCfg):
    # 开启图形渲染时才使用绘图窗口
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type: G1DebugWindow = G1DebugWindow

    observations: G1ElevHistAMPObsCfg = G1ElevHistAMPObsCfg()
