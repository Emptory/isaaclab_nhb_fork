from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
import isaaclab_nhb

if not isaaclab_nhb.HEADLESS_FLAG:
    from .ui.manager_debug_rl_window import ManagerDebugRLEnvWindow
import torch

class ManagerDebugRLEnv(ManagerBasedRLEnv):
    """继承RL环境,添加自己的debug窗口"""
    if not isaaclab_nhb.HEADLESS_FLAG:
        _window: ManagerDebugRLEnvWindow
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.action_extra_info = dict()
        self.action_extra_info["est_vel"] = torch.zeros((4096, 3), dtype=torch.float32)
        self.action_extra_info["obs_predict"] = torch.zeros((4096, 30), dtype=torch.float32) # TODO：比较抽象，需要修改
        super().__init__(cfg, render_mode, **kwargs)
        

    def step(self, action_dict: torch.Tensor) -> VecEnvStepReturn:
        """在原来step函数的基础上添加debug数据更新功能"""

        # 从字典中提取action和extra_info
        action = action_dict["action"]
        self.action_extra_info = action_dict["extra_info"]

        # 调用原来的step函数
        super().step(action)

        # 调用窗口更新函数
        if not isaaclab_nhb.HEADLESS_FLAG:
            self._window.fresh_debug_info_frame()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    

    # def step(self, action_dict: torch.Tensor) -> VecEnvStepReturn:
    #     """
    #     在原step的基础上添加了下面的功能
    #     1. 从action_dict中提取action和extra_info
    #     2. 调用debug窗口更新函数

    #     """

    #     # 从字典中提取action和extra_info
    #     action = action_dict["action"]
    #     self.action_extra_info = action_dict["extra_info"]

    #     # process actions
    #     self.action_manager.process_action(action.to(self.device))

    #     self.recorder_manager.record_pre_step()

    #     # check if we need to do rendering within the physics loop
    #     # note: checked here once to avoid multiple checks within the loop
    #     is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

    #     # perform physics stepping
    #     for _ in range(self.cfg.decimation):
    #         self._sim_step_counter += 1
    #         # set actions into buffers
    #         self.action_manager.apply_action()
    #         # set actions into simulator
    #         self.scene.write_data_to_sim()
    #         # simulate
    #         self.sim.step(render=False)
    #         self.recorder_manager.record_post_physics_decimation_step()
    #         # render between steps only if the GUI or an RTX sensor needs it
    #         # note: we assume the render interval to be the shortest accepted rendering interval.
    #         #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
    #         if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
    #             self.sim.render()
    #         # update buffers at sim dt
    #         self.scene.update(dt=self.physics_dt)

    #     # post-step:
    #     # -- update env counters (used for curriculum generation)
    #     self.episode_length_buf += 1  # step in current episode (per env)
    #     self.common_step_counter += 1  # total step (common for all envs)
    #     # -- check terminations
    #     self.reset_buf = self.termination_manager.compute()
    #     self.reset_terminated = self.termination_manager.terminated
    #     self.reset_time_outs = self.termination_manager.time_outs
    #     # -- reward computation
    #     self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    #     if len(self.recorder_manager.active_terms) > 0:
    #         # update observations for recording if needed
    #         self.obs_buf = self.observation_manager.compute()
    #         self.recorder_manager.record_post_step()

    #     # -- reset envs that terminated/timed-out and log the episode information
    #     reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    #     if len(reset_env_ids) > 0:
    #         # trigger recorder terms for pre-reset calls
    #         self.recorder_manager.record_pre_reset(reset_env_ids)

    #         # 记录reset前的amp_policy观测值，用于AMP训练
    #         # 只更新要被reset的环境的amp_policy观测，不影响其他环境和观测组
    #         if "amp" in self.observation_manager._group_obs_term_cfgs:
    #             # 计算完整观测（包含所有组）
    #             full_obs = self.observation_manager.compute()
    #             # 先保存要更新的数据，稍后在观测计算后再更新
    #             amp_policy_obs_before_reset = full_obs["amp"].clone()
    #             amp_policy_reset_ids = reset_env_ids.clone()


    #         self._reset_idx(reset_env_ids)

    #         # if sensors are added to the scene, make sure we render to reflect changes in reset
    #         if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
    #             self.sim.render()

    #         # trigger recorder terms for post-reset calls
    #         self.recorder_manager.record_post_reset(reset_env_ids)

    #     # -- update command
    #     self.command_manager.compute(dt=self.step_dt)
    #     # -- step interval events
    #     if "interval" in self.event_manager.available_modes:
    #         self.event_manager.apply(mode="interval", dt=self.step_dt)
    #     # -- compute observations
    #     # note: done after reset to get the correct observations for reset envs
    #     self.obs_buf = self.observation_manager.compute(update_history=True)

    #     # 更新reset环境的amp_policy观测值（在观测计算完成后进行，避免被覆盖）
    #     if len(reset_env_ids) > 0 and "amp" in self.observation_manager._group_obs_term_cfgs:
    #         self.obs_buf["amp"][amp_policy_reset_ids] = amp_policy_obs_before_reset[amp_policy_reset_ids]

    #     # 调用窗口更新函数
    #     if not isaaclab_nhb.HEADLESS_FLAG:
    #         self._window.fresh_debug_info_frame()

    #     # return observations, rewards, resets and extras
    #     return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
