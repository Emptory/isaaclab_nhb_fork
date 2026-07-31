import gymnasium as gym
from . import agents


gym.register(
    id="CoopG1S0-29dof-Stage0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1S0_env_cfg:CoopG1S0FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.coopG1S0_rsl_rl_ppo_cfg:CoopG1S0FlatPPORunnerCfg",
    },
)


gym.register(
    id="CoopG1S0-29dof-Stage0-Legacy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1S0_env_cfg:CoopG1S0LegacyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.coopG1S0_rsl_rl_ppo_cfg:CoopG1S0FlatPPORunnerCfg",
    },
)
