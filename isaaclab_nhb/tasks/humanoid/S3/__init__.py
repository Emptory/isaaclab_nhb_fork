

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="S3-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_env_cfg:S3RoughEnvCfg", # 指env_cfg
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3RoughPPORunnerCfg",
    },
)

gym.register(
    id="S3-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_env_cfg:S3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3FlatPPORunnerCfg",
    },
)

gym.register(
    id="S3-Symmetry-DeltaSine-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_DeltaSine_env_cfg:S3DeltaSineFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3FlatSymmetryDeltaSinePPORunnerCfg",
    },
)

gym.register(
    id="S3-Symmetry-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_env_cfg:S3RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3RoughSymmetryPPORunnerCfg",
    },
)

################################################## NoGait ##################################################
gym.register(
    id="S3-Symmetry-NoGait-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_noGait_env_cfg:S3FlatNoGaitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3NoGaitFlatPPORunnerCfg",
    },
)

################################################## EstimateNet ##################################################

gym.register(
    id="S3-EstNet-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_EstNet_env_cfg:S3RoughEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3EstNetRoughPPORunnerCfg",
    },
)

gym.register(
    id="S3-EstNet-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_EstNet_env_cfg:S3FlatEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3EstNetFlatPPORunnerCfg",
    },
)

################################################## DWAQ ##################################################

gym.register(
    id="S3-DWAQ-Rough",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_EstNet_env_cfg:S3RoughEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3DWAQRoughPPORunnerCfg",
    },
)

gym.register(
    id="S3-DWAQ-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_EstNet_env_cfg:S3FlatEstNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3DWAQFlatPPORunnerCfg",
    },
)

################################################## Mimic ##################################################

gym.register(
    id="S3-lbl3-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_lbl_env_cfg:S3lbl3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3lbl3FlatPPORunnerCfg",
    },
)

gym.register(
    id="S3-lbl2-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_lbl_env_cfg:S3lbl2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3lbl2FlatPPORunnerCfg",
    },
)

gym.register(
    id="S3-lbl1-Flat",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_lbl_env_cfg:S3lbl1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3lbl1FlatPPORunnerCfg",
    },
)


################################################# Check USD ################################################

gym.register(
    id="S3-Check-USD",
    entry_point="isaaclab_nhb.envs.manager_debug_rl_env:ManagerDebugRLEnv", # 即form{冒号前的}import{冒号后的}
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.S3_env_cfg:S3FlatEnvCfg_Check_USD", # 指env_cfg
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.S3_rsl_rl_ppo_cfg:S3FlatPPORunnerCfg",
    },
)