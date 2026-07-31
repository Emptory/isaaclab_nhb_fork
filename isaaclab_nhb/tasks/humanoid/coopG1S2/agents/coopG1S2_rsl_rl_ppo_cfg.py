import os
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from isaaclab_nhb.tasks.rl_cfg.rl_cfg import RslRlPpoActorCriticResidualCfg

from ...G1.G1_asset_cfg import G1_29DOF_JOINT_ORDER
from ...coopG1S0.agents.coopG1S0_rsl_rl_ppo_cfg import CoopG1S0FlatPPORunnerCfg
from ..coopG1S2_env import (
    VIRTUAL_FORCE_APPLICATION_POINT,
    VIRTUAL_SPRING_FORMULA,
)
from ..coopG1S2_env_cfg import S2_ARM_RESIDUAL_SCALE, VirtualTwoHandForceCfg


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

RESIDUAL_SCALE_BY_JOINT = [
    S2_ARM_RESIDUAL_SCALE
    if any(part in joint_name for part in ("shoulder", "elbow", "wrist"))
    else 0.0
    for joint_name in G1_29DOF_JOINT_ORDER
]

_DEFAULT_VIRTUAL_FORCE = VirtualTwoHandForceCfg()
FORCE_PHYSICS_CONTRACT = {
    "enabled": _DEFAULT_VIRTUAL_FORCE.enabled,
    "reference_body_name": _DEFAULT_VIRTUAL_FORCE.reference_body_name,
    "hand_body_names": list(_DEFAULT_VIRTUAL_FORCE.hand_body_names),
    "hand_anchor_offsets": [
        list(offset) for offset in _DEFAULT_VIRTUAL_FORCE.hand_anchor_offsets
    ],
    "force_control_axes": list(_DEFAULT_VIRTUAL_FORCE.force_control_axes),
    "stiffness_n_per_m": list(_DEFAULT_VIRTUAL_FORCE.stiffness_n_per_m),
    "damping_ns_per_m": list(_DEFAULT_VIRTUAL_FORCE.damping_ns_per_m),
    "max_force_n": list(_DEFAULT_VIRTUAL_FORCE.max_force_n),
    "csv_force_is_hand_on_payload": (
        _DEFAULT_VIRTUAL_FORCE.csv_force_is_hand_on_payload
    ),
    "spring_formula": VIRTUAL_SPRING_FORMULA,
    "application_point": VIRTUAL_FORCE_APPLICATION_POINT,
}


def _residual_policy(checkpoint: str) -> RslRlPpoActorCriticResidualCfg:
    return RslRlPpoActorCriticResidualCfg(
        base_policy_checkpoint=checkpoint,
        base_policy_obs_group="base_policy",
        residual_scale=RESIDUAL_SCALE_BY_JOINT,
        init_noise_std=0.1,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        append_base_action=True,
        force_estimator_obs_group="residual_policy",
        force_estimator_target_group="force_privileged",
        force_estimator_hidden_dims=[128, 64],
        force_estimator_scale=1.0,
        force_estimator_loss_coef=1.0,
        force_estimator_smooth_l1_beta=0.1,
        force_physics_contract=FORCE_PHYSICS_CONTRACT,
        policy_schema_version=5,
        observation_schema=(
            "coopG1S2_force_v5_world-spring_torso-policy_"
            "residual207_force-context12_control233_est6_actor239_critic868_"
            "force-Lxyz-Rxyz-env-on-hand"
        ),
        action_names=G1_29DOF_JOINT_ORDER,
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
    # The residual transform is already bounded.  Avoid a second wrapper-side
    # clip that would invalidate the latent-action likelihood contract.
    clip_actions = None

    obs_groups = {
        "policy": ["residual_policy", "force_context"],
        "critic": ["critic", "force_context", "force_privileged"],
        "base_policy": ["base_policy"],
        "force_privileged": ["force_privileged"],
    }

    policy = _residual_policy(
        os.environ.get("COOP_G1_S1_CHECKPOINT", DEFAULT_S1_CHECKPOINT)
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
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
