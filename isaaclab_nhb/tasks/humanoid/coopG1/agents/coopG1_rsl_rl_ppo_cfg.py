# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab_nhb.tasks.rl_cfg.rl_cfg import (
    RslRlPpoActorCriticLSTMCfg,
    RslRlPpoActorCriticTransformerCfg,
)


@configclass
class CoopG1BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "coopG1"
    run_name = "coop"
    clip_actions = 1.0
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

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
class CoopG1LSTMPPORunnerCfg(CoopG1BasePPORunnerCfg):
    run_name = "lstm"
    policy = RslRlPpoActorCriticLSTMCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_agents=2,
        token_dim=128,
        lstm_hidden_dim=128,
        lstm_num_layers=1,
    )


@configclass
class CoopG1TransformerPPORunnerCfg(CoopG1BasePPORunnerCfg):
    run_name = "transformer"
    policy = RslRlPpoActorCriticTransformerCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_agents=2,
        token_dim=128,
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_ff_dim=256,
        transformer_dropout=0.0,
    )
