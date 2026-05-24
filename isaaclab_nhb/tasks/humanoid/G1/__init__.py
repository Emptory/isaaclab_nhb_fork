import gymnasium as gym
from . import agents


################################################## Ori ##################################################

gym.register(
    id="G1-Ori-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv", # 即form{冒号前的}import{冒号后的}
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_ori_env_cfg:G1OriRoughEnvCfg", # 指env_cfg
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1OriRoughPPORunnerCfg",
    },
)

################################################## my base ##################################################


gym.register(
    id="G1-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv", # 即form{冒号前的}import{冒号后的}
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_env_cfg:G1RoughEnvCfg", # 指env_cfg
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

gym.register(
    id="G1-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

################################################## EstNet ##################################################

gym.register(
    id="G1-EstNet-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_EstNet_env_cfg:G1RoughEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1EstNetFlatPPORunnerCfg",
    },
)

gym.register(
    id="G1-EstNet-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_EstNet_env_cfg:G1FlatEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1EstNetFlatPPORunnerCfg",
    },
)


################################################## DWAQ ##################################################

gym.register(
    id="G1-DWAQ-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_EstNet_env_cfg:G1FlatEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1DWAQFlatPPORunnerCfg",
    },
)

gym.register(
    id="G1-DWAQ-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_EstNet_env_cfg:G1RoughEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1DWAQRoughPPORunnerCfg",
    },
)

################################################## AMP ##################################################

gym.register(
    id="G1-AMP-Walk-Flat",
    entry_point="isaaclab_nhb.envs.AMP_manager_based_rl_env:AMPManagerBasedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_AMP_env_cfg:G1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1AmpFlatPPORunnerCfg",
    },
)

################################################## 高程图感知 ##################################################

gym.register(
    id="G1-ElevHist-ECMM-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_elevation_env_cfg:G1ElevHistRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1ElevHistECMMRoughPPORunnerCfg",
    },
)

gym.register(
    id="G1-ElevHist-ECMM-AMP-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.G1_elevation_env_cfg:G1ElevHistAMPRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.G1_rsl_rl_ppo_cfg:G1ElevHistECMMAMPRoughPPORunnerCfg",
    },
)
