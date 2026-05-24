import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm

import isaaclab_nhb 
if not isaaclab_nhb.HEADLESS_FLAG:
    from isaaclab_nhb.envs.ui.manager_debug_rl_window import ManagerDebugRLEnvWindow
    from isaaclab_nhb.envs.manager_debug_rl_env import ManagerDebugRLEnv
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_nhb.tasks.humanoid.G1.G1_env_cfg import G1RoughEnvCfg, G1RoughObsCfg
import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from .G1_asset_cfg import G1_29DOF_CFG, G1_12DOF_JOINT_ORDER
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera 
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject

if not isaaclab_nhb.HEADLESS_FLAG:
    class G1EstNetDebugWindow(ManagerDebugRLEnvWindow):
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
            self.register_plot("vel_est_x", ["command", "estimate", "real"])
            self.register_plot("foot contact force", ["l", "r"])

        def _update_debug_data(self) -> dict:
            """更新调试数据"""
            data = {}
            
            # 速度估计值、速度命令与当前速度
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
                data["vel_est_x"] = torch.zeros(3, device=self.env.device)
            
            # 足端触地力
            l_foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link"])
            r_foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["right_ankle_roll_link"])
            l_foot_contact_sensor: ContactSensor = self.env.scene.sensors[l_foot_sensor_cfg.name]
            r_foot_contact_sensor: ContactSensor = self.env.scene.sensors[r_foot_sensor_cfg.name]
            l_foot_frc_z = l_foot_contact_sensor.data.net_forces_w[0, 7, 2]
            r_foot_frc_z = r_foot_contact_sensor.data.net_forces_w[0, 14, 2]
            data["foot contact force"] = torch.stack([l_foot_frc_z, r_foot_frc_z])
            
            return data

            # 足端高度图
            # left_raycast_sensor: RayCaster = self.env.scene.sensors["left_foot_height_scanner"]
            # right_raycast_sensor: RayCaster = self.env.scene.sensors["right_foot_height_scanner"]
            # left_raycaster_point = (left_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) - left_raycast_sensor.data.ray_hits_w[:,:,2])[0, :].reshape(21,6)
            # right_raycaster_point = (right_raycast_sensor.data.pos_w[:, 2].unsqueeze(1) - right_raycast_sensor.data.ray_hits_w[:,:,2])[0, :].reshape(21,6)
            # # 创建中间的分割线
            # zeros_insert = torch.ones((left_raycaster_point.size(0), 1), 
            #               dtype=left_raycaster_point.dtype,
            #               device=left_raycaster_point.device)

            # # 将左右传感器拼接成一张图片
            # image_to_show = torch.cat([left_raycaster_point, zeros_insert, right_raycaster_point], dim=1) * 10
            # self._add_var_to_dict("foot rayCaster", image_to_show, ["foot height image"], "image")

            

            # 相位指示器
            # gait_rew = self.env.reward_manager._class_term_cfgs[0].func
            # I_left = gait_rew.left_I[0]
            # I_right = gait_rew.right_I[0]
            # gait_I = torch.stack([I_left,I_right])
            # self._add_var_to_dict("gait I",gait_I,["I_left","I_right"], "plot")

            # # 正余弦命令
            # gait_command = self.env.command_manager.get_command("gait_command")
            # sc_command = torch.stack(
            #     [gait_command[0,3],gait_command[0,4],gait_command[0,5],gait_command[0,6]]
            # )
            # self._add_var_to_dict("sc command", sc_command, ["sin_l","cos_l","sin_r","cos_r"], "plot")

            

            


@configclass
class G1EstNetRoughObsCfg(G1RoughObsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyOneFrameCfg(ObsGroup):
        """Observations for policy group."""

        # base角速度 [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        # base重力向量 [3]
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.0, n_max=0.0))
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节位置 [29]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=G1_12DOF_JOINT_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度 [29]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=G1_12DOF_JOINT_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-1.5, n_max=1.5))
        # 上一次的动作值 [29]
        actions = ObsTerm(func=mdp.last_action)
        # 步态信息 [7]
        gait_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "gait_command"})

        # 在构造函数__init__后执行的“后构造函数”， 修改部分参数
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class PolicyCfg(ObsGroup):
        obs_history = ObsTerm(
            func=mdp_nhb.obs_history,
            params={"history_length": 5, "obs_len": 52, "obs_name": "obs_one_frame"})
            
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    obs_one_frame: PolicyOneFrameCfg = PolicyOneFrameCfg() 
    policy: PolicyCfg = PolicyCfg()


@configclass
class G1EstNetRoughRewards:

    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp,weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5})
    track_lin_vel_xy_abs = RewTerm(func=mdp_nhb.track_lin_vel_xy_yaw_frame_expabs,weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=5.0, 
        params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_expabs = RewTerm(func=mdp_nhb.track_ang_vel_z_world_expabs, weight=5.0, 
        params={"command_name": "base_velocity", "std": 0.5})

    # 机体平衡奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-5.0)
    lin_vel_z_exp = RewTerm(func=mdp_nhb.lin_vel_z_exp, weight=0.15, params={"std": 0.3})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_xy_exp = RewTerm(func=mdp_nhb.ang_vel_xy_exp, weight=0.15, params={"std": 0.45})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    flat_orientation_exp = RewTerm(func=mdp_nhb.flat_orientation_exp, weight=0.15, params={"std": 0.25})

    # 运动平滑奖励
    # action_smooth_tripple_l2 = RewTerm(func=mdp_nhb.action_smooth_tripple_l2,weight=-2e-6,params={"c_a":0.05, "c_da":2.5, "c_dda":1})
    # dof_smooth_l2 = RewTerm(func=mdp_nhb.dof_smooth_l2,weight=-2e-6,params={"c_dof_v":0.02, "c_dof_a":0.025, "c_tor":1})
    # body_smooth_l2 = RewTerm(func=mdp_nhb.body_smooth_l2,weight=-2e-6,
    #     params={
    #         "c_body_a":10, 
    #         "c_feet_a":1, 
    #         "base_asset_cfg":SceneEntityCfg("robot", body_names="torso_link"),
    #         "feet_asset_cfg":SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     }
    # )

    # 安全奖励
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])}
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    joint_vel_soft_limits = RewTerm(func=mdp_nhb.joint_vel_soft_limits, weight=-0.2, params={"soft_ratio":0.95})
    joint_tor_soft_limits = RewTerm(func=mdp_nhb.joint_tor_soft_limits, weight=-0.2, params={"soft_ratio":0.95})
    # feet_contact_force = RewTerm(func=mdp.contact_forces, weight=-0.01, 
        # params={"threshold": 480, "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")})
    # biped_distance_y_exp = RewTerm(func=mdp_nhb.biped_distance_y_l2, weight=-10, 
    #     params={"min_distance": 0.1, "max_distance": 0.35, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 步态奖励
    base_height = RewTerm(func=mdp.base_height_l2,weight=-1.0,params={"target_height": 0.7})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # stand_still = RewTerm(
    #     func=mdp_nhb.stand_still_without_cmd,
    #     weight=-0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #     },
    # )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot",joint_names=[".*_shoulder_roll_.*", ".*_shoulder_yaw_.*",".*_wrist_.*"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    )
    gait_rew = RewTerm(
        func=mdp_nhb.BipedalGaitReward,
        weight=2.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            "right_sensor_cfg":SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
            "left_asset_cfg":SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
            "right_asset_cfg":SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
            "foot_height_tar": 0.10,  # 足端高度目标值
        }
    )
    human_shoulder_trajectory_rew = RewTerm(
        func=mdp_nhb.human_shoulder_trajectory_l2,
        weight=0.5,
        params={
            "left_shoulder_cfg":SceneEntityCfg("robot",joint_names=["left_shoulder_pitch_joint"]),
            "right_shoulder_cfg":SceneEntityCfg("robot",joint_names=["right_shoulder_pitch_joint"]),
            "left_elbow_cfg":SceneEntityCfg("robot",joint_names=["left_elbow_joint"]),
            "right_elbow_cfg":SceneEntityCfg("robot",joint_names=["right_elbow_joint"]),
            "shoulder_pitch_center": 0.131,
            "elbow_center": 0.9975,
            "std": 0.3
        }
    )
    human_symmetic_rew = RewTerm(
        func=mdp_nhb.HumanSymmetricReward,
        weight=1.0,
        params={
            "std": 0.5,
            "left_asset_cfg":SceneEntityCfg("robot", joint_names=["left_.*"]),
            "right_asset_cfg":SceneEntityCfg("robot", joint_names=["right_.*"])
        }
    )  


@configclass
class G1RoughEstNetEnvCfg(G1RoughEnvCfg):
    """G1 Rough Environment Configuration with Depth Camera."""
    if not isaaclab_nhb.HEADLESS_FLAG:
        ui_window_class_type: G1EstNetDebugWindow = G1EstNetDebugWindow

    observations: G1EstNetRoughObsCfg = G1EstNetRoughObsCfg()
    # rewards: G1EstNetRoughRewards = G1EstNetRoughRewards()

    # def __post_init__(self):
    #     super().__post_init__()
    #     self.rewards.termination_penalty.weight = -1000.0
    #     self.curriculum.feet_distance_rew_weight_ratio = None

@configclass
class G1FlatEstNetEnvCfg(G1RoughEstNetEnvCfg):
    """G1 Rough Environment Configuration with Depth Camera."""
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



