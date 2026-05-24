# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from isaaclab_nhb.tasks.rl_cfg.rl_cfg import (
    RslRlPpoActorCriticEstNetCfg,
    RslRlPpoActorCriticDWAQCfg,
    RslRlPpoActorCriticECMMCfg,
    RslRlAMPCfg,
)
from isaaclab_nhb.tasks.mdp_nhb.symmetry import general_symmetry
from isaaclab_nhb.dataset import G1AmpDataCfg

##################################################Ori Configs##################################################

@configclass
class G1OriRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_ori_rough"
    run_name = "ori"
    empirical_normalization = False
    swanlab_project = "G1"
    logger = "swanlab"  # 使用swanlab记录日志
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1OriFlatPPORunnerCfg(G1OriRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_ori_flat"
        self.run_name = "ori"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

################################################## Normal Configs ##################################################

@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_rough"
    run_name = "more_track"
    logger = "swanlab"  # 使用swanlab记录日志
    swanlab_project = "G1"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
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
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=general_symmetry.compute_symmetric_states
        ),
    )

@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 10000
        self.experiment_name = "g1_flat"
        self.run_name = "29dof_test"

################################################## EstimateNet Configs ##################################################

@configclass
class G1EstNetRoughPPORunnerCfg(G1RoughPPORunnerCfg):
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_estnet_rough"
    run_name = "rough_test"
    clip_actions = 50.0
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticEstNetCfg(
        # 使用EstNet算法时的配置
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[1024, 512, 256], # 编码器隐藏层
        num_history_len=5,
        activation="elu",
    )

    def __post_init__(self):
        super().__post_init__()
        # 暂时不对历史信息进行对称化处理
        self.algorithm.symmetry_cfg = None


@configclass
class G1EstNetFlatPPORunnerCfg(G1EstNetRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 10000
        self.experiment_name = "g1_estnet_flat"
        self.run_name = "Est_L2_100"

################################################## DWAQ Configs ##################################################

@configclass
class G1DWAQRoughPPORunnerCfg(G1EstNetRoughPPORunnerCfg):
    max_iterations = 10000
    save_interval = 500
    experiment_name = "g1_dwaq_rough"
    run_name = "rough_test"
    policy = RslRlPpoActorCriticDWAQCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[1024, 512, 256],  # 编码器隐藏层
        decoder_hidden_dims=[256, 512, 1024],  # 解码器隐藏层
        num_latent=19,  # 隐向量长度，包含3维线速度
        num_history_len=5,
        activation="elu",
        VAE_beta=1.0,
    )

@configclass
class G1DWAQFlatPPORunnerCfg(G1DWAQRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 10000
        self.experiment_name = "g1_dwaq_flat"
        self.run_name = "dwaq_beta"


################################################## Elevation Configs ##################################################

@configclass
class G1ElevationRoughPPORunnerCfg(G1RoughPPORunnerCfg):
    save_interval = 500
    max_iterations = 20000
    experiment_name = "g1_elevation_rough"
    run_name = "test"

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_hidden_dims = [768, 384, 128]
        self.policy.critic_hidden_dims = [768, 384, 128]
        # 开对称不利于AMP学习
        self.algorithm.symmetry_cfg = None

################################################## ElevHistECMM Configs ##################################################

@configclass
class G1ElevHistECMMRoughPPORunnerCfg(G1ElevationRoughPPORunnerCfg):
    """
    观测值:
    policy: prop+elevation历史值
    critic: priv+elevation历史值
    
    网络结构:
    policy: (CNN+MLP)->MLP
    critic: (CNN+MLP)->MLP
    """
    save_interval = 500
    max_iterations = 50000
    experiment_name = "G1_ElevHist_ECMM_rough"
    run_name = "rename_ECMM_delete_size_2cm"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "height_scan_policy": ["height_scan_policy"],
        "height_scan_critic": ["height_scan_critic"],
    }
    
    policy = RslRlPpoActorCriticECMMCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # 特征维度配置
        vision_feature_dim=64,     # 2DCNN输出64维特征
        actor_mlp_feature_dim=64,  # Actor MLP特征维度
        # Actor 2DCNN配置
        actor_cnn_hidden_dims=[8, 16],
        actor_cnn_kernel_sizes=[3, 3],
        actor_cnn_strides=[2, 2],
        # Critic 2DCNN配置（使用相同配置）
        critic_cnn_hidden_dims=None,  # 使用Actor的配置
        critic_cnn_kernel_sizes=None,  # 使用Actor的配置
        critic_cnn_strides=None,  # 使用Actor的配置
        # Actor MLP特征提取器配置
        actor_mlp_hidden_dims=[128],
        # Actor/Critic网络配置
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class G1ElevHistECMMAMPRoughPPORunnerCfg(G1ElevHistECMMRoughPPORunnerCfg):
    """
    与上一个一致，只是加上了AMP
    
    """
    experiment_name = "G1_ElevHist_ECMM_rough"
    run_name = "new_edge_pal_test2_AMP"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "height_scan_policy": ["height_scan_policy"],
        "height_scan_critic": ["height_scan_critic"],
        "amp": ["amp"],
    }
    
    amp_cfg = RslRlAMPCfg(
        reward_coef=0.35,
        task_reward_lerp=0.6,
        discr_hidden_dims=[256, 128, 64],
        discr_update_decimation=1,
        discr_learning_rate=5e-4,
        discr_normalize=False,
        num_preload_transitions=200000,
        amp_motion_files=G1AmpDataCfg().motion_expert_path,
    )

################################################## AMP Configs ##################################################

@configclass
class G1AmpRoughPPORunnerCfg(G1RoughPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "g1_amp_rough"
    run_name = "ampRough_test"

    amp_cfg = RslRlAMPCfg(
        reward_coef=0.35,
        task_reward_lerp=0.6,
        discr_hidden_dims=[1024, 512, 256],
        discr_update_decimation=1,
        discr_learning_rate=5e-4,
        discr_normalize=False,
        num_preload_transitions=200000,
        amp_motion_files=G1AmpDataCfg().motion_expert_path,
    )

    # 开对称效果反而变差了
    def __post_init__(self):
        super().__post_init__()
        # 暂时不对历史信息进行对称化处理
        self.algorithm.symmetry_cfg = None
        self.logger = "tensorboard"

@configclass
class G1AmpFlatPPORunnerCfg(G1AmpRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 20000
        self.experiment_name = "g1_amp_flat"
        self.run_name = "new_reward_action_-1e-4_random"
