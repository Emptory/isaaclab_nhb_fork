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

from .S3_asset_cfg import S3_12DOF_CFG, S3_12DOF_JOINT_LOWER_ORDER, S3_12DOF_FIX_BASE_CFG, S3_22DOF_JOINT_ORDER, S3_22DOF_CFG, S3_12DOF_JOINT_UPPER_ORDER
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from dataclasses import MISSING
from .S3_env_cfg import S3RoughObsCfg, S3RoughEnvCfg, S3FlatEnvCfg


@configclass
class S3EstNetRoughObsCfg():
    """Observation specifications for the MDP."""

    @configclass
    class PolicyOneFrameCfg(ObsGroup):
        """Observations for policy group."""
        # base角速度 [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.02, n_max=0.02))
        # base重力向量 [3]
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.02, n_max=0.02))
        # 关节位置 [12]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=S3_12DOF_JOINT_LOWER_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度 [12]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg("robot", 
                                                            joint_names=S3_12DOF_JOINT_LOWER_ORDER,
                                                            preserve_order=True)},
                                noise=Unoise(n_min=-1.5, n_max=1.5))
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
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
    class PolicyCfg(ObsGroup):
        obs_history = ObsTerm(
            func=mdp_nhb.obs_history,
            params={"history_length": 5, "obs_len": 52, "obs_name": "obs_one_frame"})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""
        # 机体线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # base重力向量
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
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
        # 线速度命令 [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
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
    
    obs_one_frame: PolicyOneFrameCfg = PolicyOneFrameCfg() 
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class S3RoughEstNetEnvCfg(S3RoughEnvCfg):
    """S3 Rough Environment Configuration with Depth Camera."""

    observations: S3EstNetRoughObsCfg = S3EstNetRoughObsCfg()


@configclass
class S3FlatEstNetEnvCfg(S3RoughEstNetEnvCfg):
    """S3 Rough Environment Configuration with Depth Camera."""
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