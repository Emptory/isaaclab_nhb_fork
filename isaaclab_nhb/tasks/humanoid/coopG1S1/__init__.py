import gymnasium as gym

from . import agents


gym.register(
    id="CoopG1S1-29dof-HoldBox",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1S1_env_cfg:CoopG1S1HoldBoxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.coopG1S1_rsl_rl_ppo_cfg:CoopG1S1HoldBoxPPORunnerCfg",
    },
)
