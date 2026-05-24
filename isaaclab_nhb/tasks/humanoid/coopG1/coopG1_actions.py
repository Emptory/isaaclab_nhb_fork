from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class EMAJointPositionAction(JointPositionAction):
    """Joint position action with EMA smoothing on the final joint target."""

    cfg: "EMAJointPositionActionCfg"

    def __init__(self, cfg: "EMAJointPositionActionCfg", env):
        super().__init__(cfg, env)
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {cfg.alpha}.")
        self._alpha = cfg.alpha
        self._prev_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        targets = self._alpha * self._processed_actions + (1.0 - self._alpha) * self._prev_targets
        targets = torch.clamp(
            targets,
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        self._processed_actions[:] = targets
        self._prev_targets[:] = targets

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            super().reset(slice(None))
            self._prev_targets[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            if not torch.is_tensor(env_ids):
                env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            super().reset(env_ids)
            self._prev_targets[env_ids] = self._asset.data.joint_pos[env_ids[:, None], self._joint_ids].view(
                len(env_ids), -1
            )


@configclass
class EMAJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = EMAJointPositionAction
    alpha: float = 0.15
