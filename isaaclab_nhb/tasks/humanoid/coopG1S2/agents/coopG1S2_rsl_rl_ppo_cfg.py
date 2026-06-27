import os
from pathlib import Path

from isaaclab.utils import configclass

from isaaclab_nhb.tasks.rl_cfg.rl_cfg import RslRlPpoActorCriticResidualCfg

from ...coopG1S0.agents.coopG1S0_rsl_rl_ppo_cfg import CoopG1S0FlatPPORunnerCfg


DEFAULT_S1_CHECKPOINT = str(
    Path(__file__).resolve().parents[4]
    / "logs/rsl_rl/coopG1S1/2026-06-22_11-26-35_s1_base_velocity_precise/model_12998.pt"
)
DEFAULT_S1_LEGACY_57998_CHECKPOINT = str(
    Path(__file__).resolve().parents[4]
    / "logs/rsl_rl/coopG1S1"
    / "2026-06-14_21-52-04_s1_from2026-06-10_07-38-467999/model_57998.pt"
)
DEFAULT_S1_LEGACY_62997_CHECKPOINT = str(
    Path(__file__).resolve().parents[4]
    / "logs/rsl_rl/coopG1S1"
    / "2026-06-14_21-51-53_s1_from2026-06-10_17-12-2712998/model_62997.pt"
)


def _residual_policy(checkpoint: str) -> RslRlPpoActorCriticResidualCfg:
    return RslRlPpoActorCriticResidualCfg(
        base_policy_checkpoint=checkpoint,
        base_policy_obs_group="base_policy",
        residual_scale=0.1,
        init_noise_std=0.1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class CoopG1S2FixedPayloadPPORunnerCfg(CoopG1S0FlatPPORunnerCfg):
    experiment_name = "coopG1S2"
    run_name = "fixed_payload"


@configclass
class CoopG1S2ResidualPPORunnerCfg(CoopG1S0FlatPPORunnerCfg):
    experiment_name = "coopG1S2"
    run_name = "residual_s1"
    init_at_random_ep_len = False

    obs_groups = {
        "policy": ["residual_policy"],
        "critic": ["critic"],
        "base_policy": ["base_policy"],
    }

    policy = _residual_policy(
        os.environ.get("COOP_G1_S1_CHECKPOINT", DEFAULT_S1_CHECKPOINT)
    )


@configclass
class CoopG1S2Legacy57998ResidualPPORunnerCfg(CoopG1S2ResidualPPORunnerCfg):
    """S2 residual training with the first 515-D legacy S1 actor."""

    run_name = "residual_s1_legacy515_model57998"
    policy = _residual_policy(
        os.environ.get("COOP_G1_S1_LEGACY_57998_CHECKPOINT", DEFAULT_S1_LEGACY_57998_CHECKPOINT)
    )


@configclass
class CoopG1S2Legacy62997ResidualPPORunnerCfg(CoopG1S2ResidualPPORunnerCfg):
    """S2 residual training with the second 515-D legacy S1 actor."""

    run_name = "residual_s1_legacy515_model62997"
    policy = _residual_policy(
        os.environ.get("COOP_G1_S1_LEGACY_62997_CHECKPOINT", DEFAULT_S1_LEGACY_62997_CHECKPOINT)
    )
