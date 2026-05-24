

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Pegasus-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", # 即form{冒号前的}import{冒号后的}
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Pegasus_rough_env_cfg:PegasusRoughEnvCfg", # 指env_cfg
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Pegasus_rsl_rl_ppo_cfg:PegasusRoughPPORunnerCfg",
    },
)

gym.register(
    id="Pegasus-Rough-Play",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Pegasus_rough_env_cfg:PegasusRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Pegasus_rsl_rl_ppo_cfg:PegasusRoughPPORunnerCfg",
    },
)

gym.register(
    id="Pegasus-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Pegasus_flat_env_cfg:PegasusFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Pegasus_rsl_rl_ppo_cfg:PegasusFlatPPORunnerCfg",
    },
)


gym.register(
    id="Pegasus-Flat-Play",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Pegasus_flat_env_cfg:PegasusFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Pegasus_rsl_rl_ppo_cfg:PegasusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Pegasus-Symmetry-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Pegasus_flat_env_cfg:PegasusFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Pegasus_rsl_rl_ppo_cfg:PegasusFlatSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="Pegasus-Symmetry-Flat-Play",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Pegasus_flat_env_cfg:PegasusFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Pegasus_rsl_rl_ppo_cfg:PegasusFlatSymmetryPPORunnerCfg",
    },
)