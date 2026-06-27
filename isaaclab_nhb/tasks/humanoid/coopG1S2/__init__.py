import gymnasium as gym

from . import agents


gym.register(
    id="CoopG1S2-29dof-FixedPayload",
    entry_point="isaaclab_nhb.tasks.humanoid.coopG1S2.coopG1S2_env:CoopG1S2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1S2_env_cfg:CoopG1S2FixedPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.coopG1S2_rsl_rl_ppo_cfg:CoopG1S2FixedPayloadPPORunnerCfg",
        "rsl_rl_residual_cfg_entry_point": (
            f"{agents.__name__}.coopG1S2_rsl_rl_ppo_cfg:CoopG1S2ResidualPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CoopG1S2-29dof-Residual-BaseVelocity530",
    entry_point="isaaclab_nhb.tasks.humanoid.coopG1S2.coopG1S2_env:CoopG1S2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1S2_env_cfg:CoopG1S2FixedPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.coopG1S2_rsl_rl_ppo_cfg:CoopG1S2ResidualPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CoopG1S2-29dof-Residual-Legacy515-57998",
    entry_point="isaaclab_nhb.tasks.humanoid.coopG1S2.coopG1S2_env:CoopG1S2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1S2_env_cfg:CoopG1S2LegacyFixedPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.coopG1S2_rsl_rl_ppo_cfg:"
            "CoopG1S2Legacy57998ResidualPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CoopG1S2-29dof-Residual-Legacy515-62997",
    entry_point="isaaclab_nhb.tasks.humanoid.coopG1S2.coopG1S2_env:CoopG1S2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1S2_env_cfg:CoopG1S2LegacyFixedPayloadEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.coopG1S2_rsl_rl_ppo_cfg:"
            "CoopG1S2Legacy62997ResidualPPORunnerCfg"
        ),
    },
)
