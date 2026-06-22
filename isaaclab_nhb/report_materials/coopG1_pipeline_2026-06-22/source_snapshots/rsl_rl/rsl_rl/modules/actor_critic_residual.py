# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from .actor_critic import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCriticResidual(ActorCritic):
    """Frozen base policy plus a trainable residual actor-critic."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        env_cfg=None,
        alg_cfg: dict | None = None,
        base_policy_checkpoint: str = "",
        base_policy_obs_group: str = "base_policy",
        residual_scale: float = 0.1,
        base_actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        base_critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        base_activation: str = "elu",
        base_actor_obs_normalization: bool = True,
        base_critic_obs_normalization: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            env_cfg=env_cfg,
            alg_cfg=alg_cfg,
            **kwargs,
        )
        if self.state_dependent_std:
            raise ValueError("ActorCriticResidual does not support state-dependent action standard deviation.")
        if not base_policy_checkpoint:
            raise ValueError(
                "A trained S1 checkpoint is required. Set COOP_G1_S1_CHECKPOINT before starting S2 residual training."
            )
        if not os.path.isfile(base_policy_checkpoint):
            raise FileNotFoundError(f"S1 checkpoint does not exist: {base_policy_checkpoint}")

        if base_policy_obs_group not in obs:
            raise KeyError(
                f"Base observation group '{base_policy_obs_group}' is missing. Available groups: {list(obs.keys())}"
            )
        if len(obs[base_policy_obs_group].shape) != 2:
            raise ValueError("The frozen base actor requires a flattened 1D observation group.")

        self.residual_scale = residual_scale
        self.base_policy_obs_group = base_policy_obs_group
        num_base_obs = obs[base_policy_obs_group].shape[-1]
        self.base_actor = MLP(num_base_obs, num_actions, base_actor_hidden_dims, base_activation)
        self.base_actor_obs_normalizer = (
            EmpiricalNormalization(num_base_obs) if base_actor_obs_normalization else nn.Identity()
        )

        checkpoint = torch.load(base_policy_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        actor_state = {
            key.removeprefix("actor."): value for key, value in state_dict.items() if key.startswith("actor.")
        }
        if not actor_state:
            raise KeyError(f"Checkpoint contains no actor parameters: {base_policy_checkpoint}")
        self.base_actor.load_state_dict(actor_state)
        if base_actor_obs_normalization:
            normalizer_state = {
                key.removeprefix("actor_obs_normalizer."): value
                for key, value in state_dict.items()
                if key.startswith("actor_obs_normalizer.")
            }
            if not normalizer_state:
                raise KeyError(f"Checkpoint contains no actor observation normalizer: {base_policy_checkpoint}")
            self.base_actor_obs_normalizer.load_state_dict(normalizer_state)
        self.base_actor.requires_grad_(False)
        self.base_actor_obs_normalizer.requires_grad_(False)
        self.base_actor.eval()
        self.base_actor_obs_normalizer.eval()

        # Start residual training from the frozen S1 behavior.
        for layer in reversed(self.actor):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_actor.eval()
        self.base_actor_obs_normalizer.eval()
        return self

    def _action_mean(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            base_obs = self.base_actor_obs_normalizer(obs[self.base_policy_obs_group])
            base_action = self.base_actor(base_obs)

        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        residual_action = self.residual_scale * torch.tanh(self.actor(actor_obs))
        return base_action + residual_action

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        mean = self._action_mean(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        return self._action_mean(obs), self.extra_info

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return {"optimizer": torch.optim.Adam(trainable_parameters, lr=learning_rate)}
