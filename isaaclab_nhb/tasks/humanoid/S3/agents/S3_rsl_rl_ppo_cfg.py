# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,RslRlSymmetryCfg
from isaaclab_nhb.tasks.rl_cfg.rl_cfg import RslRlPpoActorCriticDWAQCfg, RslRlPpoActorCriticEstNetCfg, RslRlPpoActorCriticDeltaSineCfg
from isaaclab_nhb.tasks.mdp_nhb.symmetry import general_symmetry

# TODO: 弄懂每个参数的含义
@configclass
class S3RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "S3_rough"
    run_name = "tracking_3"
    logger = "swanlab"  # 使用swanlab记录日志
    swanlab_project = "S3"
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
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=True, data_augmentation_func=general_symmetry.compute_symmetric_states
        ),
    )


@configclass
class S3FlatPPORunnerCfg(S3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 10000  
        self.experiment_name = "S3_flat"
        self.run_name = "push1_5-1_0_delay3_footY"

# @configclass
# class S3FlatSymmetryDeltaSinePPORunnerCfg(S3FlatSymmetryPPORunnerCfg):
#     policy = RslRlPpoActorCriticDeltaSineCfg(
#         noise_std_type="log",
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         deltasine_hidden_dims=[512, 256, 128],
#         activation="elu",
#         num_history_len=5,
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.008,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#         symmetry_cfg = RslRlSymmetryCfg(
#             use_data_augmentation=True, use_mirror_loss=True, data_augmentation_func=S3_symmetry_DeltaSine.S3_compute_symmetric_states_deltasine
#         ),
#     )
#     def __post_init__(self):
#         super().__post_init__()
#         self.save_interval = 500
#         self.max_iterations = 15000
#         self.experiment_name = "S3_flat_symmetry_deltaSine"
#         self.run_name = "symmetric_deltaSine_test"

################################################## lbl Configs##################################################

@configclass
class S3lbl3FlatPPORunnerCfg(S3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 30000  
        self.experiment_name = "S3_lbl3_flat"
        self.run_name = "lbl3_test"
        self.algorithm.symmetry_cfg = None

@configclass
class S3lbl2FlatPPORunnerCfg(S3lbl3FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "S3_lbl2_flat"
        self.run_name = "lbl2_test"

@configclass
class S3lbl1FlatPPORunnerCfg(S3lbl2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "S3_lbl1_flat"
        self.run_name = "lbl1_test"

##################################################NoGait Configs##################################################
@configclass
class S3NoGaitFlatPPORunnerCfg(S3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 30000  
        self.experiment_name = "S3_flat_noGait"
        self.run_name = "symmetric_noGait"

##################################################EstimateNet Configs##################################################

@configclass
class S3EstNetRoughPPORunnerCfg(S3RoughPPORunnerCfg):
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
        self.save_interval = 500
        self.max_iterations = 15000
        self.experiment_name = "S3_EstNet_rough"
        self.run_name = "est_rough_test"
        self.algorithm.symmetry_cfg = None

@configclass
class S3EstNetFlatPPORunnerCfg(S3EstNetRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 150000
        self.experiment_name = "S3_EstNet_flat"
        self.run_name = "estNetTest-velMarmalize-no1000"

################################################## DWAQ Configs ##################################################

@configclass
class S3DWAQRoughPPORunnerCfg(S3EstNetRoughPPORunnerCfg):
    max_iterations = 15000
    save_interval = 500
    experiment_name = "s3_dwaq_rough"
    run_name = "rough_test"

    policy = RslRlPpoActorCriticDWAQCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[1024, 512, 256], # 编码器隐藏层
        decoder_hidden_dims=[256, 512, 1024], # 解码器隐藏层
        num_decode=30, # 要预测的观测维度数量
        num_latent=19, # 隐向量长度，包含3维线速度 
        num_history_len=5,
        activation="elu",
        VAE_beta=1.0,
        use_adaboot=False, # 目前使用adaboot训不出来 
    )

@configclass
class S3DWAQFlatPPORunnerCfg(S3DWAQRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 15000
        self.experiment_name = "s3_dwaq_flat"
        self.run_name = "DWAQ_footH0_15"