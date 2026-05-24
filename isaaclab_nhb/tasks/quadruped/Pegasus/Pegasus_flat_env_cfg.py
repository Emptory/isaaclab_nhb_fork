# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .Pegasus_rough_env_cfg import PegasusRoughEnvCfg

import isaaclab_nhb 

##
# Pre-defined configs
##

@configclass
class PegasusFlatEnvCfg(PegasusRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class PegasusFlatEnvCfg_PLAY(PegasusFlatEnvCfg):
    
    # 开启图形渲染时才使用绘图窗口
    if not isaaclab_nhb.HEADLESS_FLAG:
        from isaaclab_nhb.tasks.quadruped.Pegasus.Pegasus_rough_env_cfg import PegasusDebugWindow
        ui_window_class_type:PegasusDebugWindow = PegasusDebugWindow

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
