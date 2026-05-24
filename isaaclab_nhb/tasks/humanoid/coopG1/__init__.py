import gymnasium as gym
from . import agents

# NOTE:
# - 这里严格遵循本仓库 G1/S3 的写法：只做任务注册，不在 import 阶段加载 env/env_cfg。
# - 这样可以避免 Python 循环导入（尤其是 G1_asset_cfg 需要从 isaaclab_nhb 读取路径常量）。

gym.register(
    id="Humanoid-CoopG1-LSTM-v0",
    entry_point="isaaclab_nhb.tasks.humanoid.coopG1.coopG1_env:CoopG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1_env_cfg:CoopG1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.coopG1_rsl_rl_ppo_cfg:CoopG1LSTMPPORunnerCfg",
    },
)

gym.register(
    id="Humanoid-CoopG1-Transformer-v0",
    entry_point="isaaclab_nhb.tasks.humanoid.coopG1.coopG1_env:CoopG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.coopG1_env_cfg:CoopG1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.coopG1_rsl_rl_ppo_cfg:CoopG1TransformerPPORunnerCfg",
    },
)
