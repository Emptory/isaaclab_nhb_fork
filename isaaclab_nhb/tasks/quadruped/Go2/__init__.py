

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv", # 即form{冒号前的}import{冒号后的}
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_env_cfg:Go2RoughEnvCfg", # 指env_cfg
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)


gym.register(
    id="Go2-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-lbl1-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_lbl_env_cfg:Go2lbl1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2lbl1PPORunnerCfg",
    },
)

gym.register(
    id="Go2-lbl2-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_lbl_env_cfg:Go2lbl2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2lbl2PPORunnerCfg",
    },
)

gym.register(
    id="Go2-lbl3-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_lbl_env_cfg:Go2lbl3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2lbl3PPORunnerCfg",
    },
)

################################################## Elevation Net Mode12L ##################################################

gym.register(
    id="Go2-Elevation-Net-Mode12L-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_elevation_net_mode12_env_cfg:Go2ElevationNetMode12RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2ElevationNetMode12LRoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Elevation-Net-Mode12L-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_elevation_net_mode12_env_cfg:Go2ElevationNetMode12FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2ElevationNetMode12LFlatPPORunnerCfg",
    },
)

################################################## Elevation Net Mode12P2 ##################################################

gym.register(
    id="Go2-Elevation-Net-Mode12P2-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_elevation_net_mode12P2_env_cfg:Go2ElevationNetMode12P2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2ElevationNetMode12P2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Elevation-Net-Mode12P2-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2_elevation_net_mode12P2_env_cfg:Go2ElevationNetMode12P2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.Go2_rsl_rl_ppo_cfg:Go2ElevationNetMode12P2FlatPPORunnerCfg",
    },
)
