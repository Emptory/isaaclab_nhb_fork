from isaaclab.utils import configclass

from ...coopG1S0.agents.coopG1S0_rsl_rl_ppo_cfg import CoopG1S0FlatPPORunnerCfg


@configclass
class CoopG1S1HoldBoxPPORunnerCfg(CoopG1S0FlatPPORunnerCfg):
    experiment_name = "coopG1S1"
    run_name = "holdbox"
    save_interval = 1000
