import os
from pathlib import Path

from isaaclab.utils import configclass

from isaaclab_nhb.tasks.rl_cfg.rl_cfg import RslRlPpoActorCriticResidualCfg

from ...coopG1S0.agents.coopG1S0_rsl_rl_ppo_cfg import CoopG1S0FlatPPORunnerCfg


DEFAULT_S1_CHECKPOINT = str(
    Path(__file__).resolve().parents[4]
    / "logs/rsl_rl/coopG1S1/2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k/model_9999.pt"
)


@configclass
class CoopG1S2FixedPayloadPPORunnerCfg(CoopG1S0FlatPPORunnerCfg):
    experiment_name = "coopG1S2"
    run_name = "fixed_payload"


@configclass
class CoopG1S2ResidualPPORunnerCfg(CoopG1S0FlatPPORunnerCfg):
    experiment_name = "coopG1S2"
    run_name = "residual_s1"

    obs_groups = {
        "policy": ["residual_policy"],
        "critic": ["critic"],
        "base_policy": ["base_policy"],
    }

    policy = RslRlPpoActorCriticResidualCfg(
        base_policy_checkpoint=os.environ.get("COOP_G1_S1_CHECKPOINT", DEFAULT_S1_CHECKPOINT),
        base_policy_obs_group="base_policy",
        residual_scale=0.1,
        init_noise_std=0.1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
