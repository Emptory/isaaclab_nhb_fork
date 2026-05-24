# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from isaaclab_nhb.tasks.mdp_nhb.symmetry import Go2_symmetry

# TODO: 弄懂每个参数的含义
@configclass
class Go2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "Go2_rough"
    run_name = "base-height-2GPU"
    empirical_normalization = True
    logger = "swanlab"  # 使用swanlab记录日志
    swanlab_project = "Go2"
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
class Go2FlatPPORunnerCfg(Go2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "Go2_flat"
        self.run_name = "gait_1.5"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

@configclass
class Go2FlatSymmetryPPORunnerCfg(Go2FlatPPORunnerCfg):
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
            use_data_augmentation=True, use_mirror_loss=True, data_augmentation_func=Go2_symmetry.Go2_compute_symmetric_states
        ),
    )
    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 500
        self.max_iterations = 3000
        self.experiment_name = "Go2_flat_symmetry"
        self.run_name = "symmetric_dev1_3"