# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from isaaclab_nhb.tasks.mdp_nhb.symmetry import general_symmetry

from isaaclab_nhb.tasks.rl_cfg.rl_cfg import RslRlPpoActorCriticElevationNetMode12LCfg, RslRlPpoActorCriticElevationNetMode12P2Cfg

# TODO: 弄懂每个参数的含义
@configclass
class Go2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 500
    experiment_name = "Go2_rough"
    run_name = "robot_lab_reward_noTermi"
    logger = "swanlab"  # 使用swanlab记录日志
    swanlab_project = "Go2"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg = RslRlSymmetryCfg(
        #     use_data_augmentation=True, use_mirror_loss=True, data_augmentation_func=general_symmetry.compute_symmetric_states
        # ),
    )


@configclass
class Go2FlatPPORunnerCfg(Go2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "Go2_flat"
        self.run_name = "test"
        # self.policy.actor_hidden_dims = [256, 128, 128]
        # self.policy.critic_hidden_dims = [256, 128, 128]


@configclass
class Go2lbl1PPORunnerCfg(Go2RoughPPORunnerCfg):
    """Go2lbl1训练配置 - 无步态无mimic"""
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Go2_lbl1"
        self.run_name = "lbl1-1_25Hz"


@configclass
class Go2lbl2PPORunnerCfg(Go2RoughPPORunnerCfg):
    """Go2lbl2训练配置 - 有步态无mimic"""
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Go2_lbl2"
        self.run_name = "lbl2-1_25Hz"


@configclass
class Go2lbl3PPORunnerCfg(Go2RoughPPORunnerCfg):
    """Go2lbl3训练配置 - 有步态有mimic"""
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Go2_lbl3"
        self.run_name = "lbl3-delay"



@configclass
class Go2ElevationNetMode12LRoughPPORunnerCfg(Go2RoughPPORunnerCfg):
    """Mode12L配置：R(2+1)D处理特权历史高程图 + MLP处理本体观测
    
    核心特性：
    - Critic: 单帧本体特权观测 -> MLP特征 + 特权高程图 -> R(2+1)D特征 -> Critic网络
    - Actor: 单帧本体观测 -> MLP特征 + 历史高程图 -> R(2+1)D特征 -> Actor网络
    """
    save_interval = 500
    max_iterations = 20000
    experiment_name = "go2_elevation_net_mode12L_rough"
    run_name = "mode12L-23Stairs-symmetry-test"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "height_scan_policy": ["height_scan_policy"],
        "height_scan_critic": ["height_scan_critic"],
    }

    policy = RslRlPpoActorCriticElevationNetMode12LCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # 高程图配置
        elevation_sampled_frames=5,   # 采样后5帧
        vision_spatial_size=(32, 32),
        # 特征维度配置
        vision_feature_dim=32,     # R(2+1)D输出32维特征
        actor_mlp_feature_dim=64,  # Actor MLP特征维度
        critic_mlp_feature_dim=64, # Critic MLP特征维度
        # R(2+1)D配置
        # r21d_hidden_dims=[6, 12, 24],
        # r21d_kernel_sizes=[3, 3, 3],
        r21d_hidden_dims=[8, 16],
        r21d_kernel_sizes=[3, 3],
        # MLP特征提取器配置
        mlp_extractor_hidden_dims=[128],
        # Actor/Critic网络配置
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.01,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    #     symmetry_cfg = RslRlSymmetryCfg(
    #         use_data_augmentation=True, use_mirror_loss=True, data_augmentation_func=general_symmetry.compute_symmetric_states
    #     ),
    # )



@configclass
class Go2ElevationNetMode12LFlatPPORunnerCfg(Go2ElevationNetMode12LRoughPPORunnerCfg):
    save_interval = 500
    max_iterations = 20000
    experiment_name = "go2_elevation_net_mode12L_flat"
    run_name = "mode12L-flat"



@configclass
class Go2ElevationNetMode12P2RoughPPORunnerCfg(Go2RoughPPORunnerCfg):
    """Mode12P2配置：VAE架构的编码器-解码器网络
    
    核心特性：
    - Critic: 本体观测 -> MLP + 高程图 -> R(2+1)D -> Value
    - Encoder (VAE): 
      * 高程图 -> R(2+1)D -> μ_c, σ_c -> z_c
      * 本体历史 -> MLP -> v̂_t + z_p
    - Decoder:
      * z_c -> 重建无噪声高程图
      * [z_c + z_p + v̂_t] -> 重建下一帧本体观测
    - Actor: [单帧本体 + v̂_t + z_c + z_p] -> 动作
    """
    save_interval = 500
    max_iterations = 20000
    experiment_name = "go2_elevation_net_mode12P2_rough"
    run_name = "mode12P2-VAE-bigger"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "height_scan_policy": ["height_scan_policy"],
        "height_scan_critic": ["height_scan_critic"],
        "obs_one_frame": ["obs_one_frame"],
    }

    policy = RslRlPpoActorCriticElevationNetMode12P2Cfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # 高程图配置
        elevation_sampled_frames=5,   # 采样后5帧
        vision_spatial_size=(32, 32),
        # 特征维度配置
        vision_feature_dim=64,     # R(2+1)D输出64维特征
        actor_mlp_feature_dim=64,  # Actor MLP特征维度
        critic_mlp_feature_dim=64, # Critic MLP特征维度
        # R(2+1)D配置
        r21d_hidden_dims=[8, 16],
        r21d_kernel_sizes=[3, 3],
        # critic MLP特征提取器配置
        mlp_extractor_hidden_dims=[128],
        # Actor/Critic网络配置
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        # 编码器参数（Mode12P2特有）
        encoder_hidden_dims=[256, 128],
        latent_dim=32,  # 高程图VAE隐变量维度
        vel_dim=3,      # 速度维度
        proprio_latent_dim=32,   # 本体隐变量维度
        # 解码器参数（Mode12P2特有）
        decoder_hidden_dims=[128, 256],
        num_decode=30,  # 重建的观测维度
        # KL权重配置
        kl_weight=0.1,  # KL散度权重，建议0.1-0.2使KL贡献30-40%总损失
    )
    # TODO:  仔细分配超参数维度

@configclass
class Go2ElevationNetMode12P2FlatPPORunnerCfg(Go2ElevationNetMode12P2RoughPPORunnerCfg):
    save_interval = 500
    max_iterations = 20000
    experiment_name = "go2_elevation_net_mode12P2_flat"
    run_name = "mode12P2-flat-test"


