from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse


VIRTUAL_SPRING_FORMULA = (
    "F_env_on_hand=Kp*(x_target-x_hand)+Kd*(v_target-v_hand)"
)
VIRTUAL_FORCE_APPLICATION_POINT = "actual_palm_anchor_world"
S2_ARM_ACTION_START_INDEX = 15


class VirtualTwoHandForceManager:
    r"""Apply and measure paper-style virtual spring forces at both palms.

    At every physics substep, each controlled translation axis uses

    .. math::

        F_\mathrm{env\rightarrow hand}
        = K_p (x_s - x_g) + K_d (v_s - v_g).

    ``x_s``/``v_s`` are the moving offline-trajectory anchor and ``x_g``/
    ``v_g`` are the current palm-surface anchor state. The force command is
    independent from this spring equation and appears only in observations,
    rewards, and diagnostics. The policy must create the spring deflection
    needed to make the measured virtual force match that command.

    This manager tracks translational force only. Passing a non-COM
    application point lets PhysX create the physically implied ``r x F``
    moment; that is not moment tracking.
    """

    def __init__(self, cfg, env: "CoopG1S2Env", command_term) -> None:
        self.cfg = cfg
        self._enabled = bool(cfg.enabled)
        self._env = env
        self._command_term = command_term
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._hand_body_ids = self._robot.find_bodies(
            cfg.hand_body_names, preserve_order=True
        )[0]
        if len(self._hand_body_ids) != 2:
            raise ValueError(
                f"Virtual spring requires two hand bodies, found {len(self._hand_body_ids)}."
            )
        self._torso_body_id = self._robot.find_bodies(cfg.reference_body_name)[0][0]

        self._anchor_offsets_b = self._vector(
            cfg.hand_anchor_offsets, (2, 3), "hand_anchor_offsets"
        )
        if not torch.allclose(
            self._anchor_offsets_b,
            command_term.hand_anchor_offsets,
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "Virtual spring and hand-reference command use different palm "
                "anchor offsets."
            )
        self._force_axis_mask_n = self._vector(
            cfg.force_control_axes, (3,), "force_control_axes"
        )
        if torch.any((self._force_axis_mask_n != 0.0) & (self._force_axis_mask_n != 1.0)):
            raise ValueError("force_control_axes entries must be binary (0.0 or 1.0).")
        if self._enabled and not bool(torch.any(self._force_axis_mask_n).item()):
            raise ValueError("At least one virtual-force control axis must be enabled.")
        if not self._enabled:
            # A disabled spring is a coherent all-position-control ablation:
            # there is no target/applied force and no kinematic axis is
            # silently removed from the task.
            self._force_axis_mask_n.zero_()
        self._position_axis_mask_n = 1.0 - self._force_axis_mask_n

        self._kp = self._vector(cfg.stiffness_n_per_m, (3,), "stiffness_n_per_m")
        self._kd = self._vector(cfg.damping_ns_per_m, (3,), "damping_ns_per_m")
        self._max_force = self._vector(cfg.max_force_n, (3,), "max_force_n")
        if torch.any(self._kp <= 0.0) or torch.any(self._kd < 0.0):
            raise ValueError("Virtual spring stiffness must be positive and damping non-negative.")
        if torch.any(self._max_force <= 0.0):
            raise ValueError("Virtual spring max_force_n must be positive.")

        shape = (env.num_envs, 2, 3)
        self.target_force_robot_w = torch.zeros(shape, device=env.device)
        self.actual_virtual_force_robot_w = torch.zeros(shape, device=env.device)
        self.spring_position_error_n = torch.zeros(shape, device=env.device)
        self.spring_velocity_error_n = torch.zeros(shape, device=env.device)
        self.force_error_robot_w = torch.zeros(shape, device=env.device)
        self.force_clamped = torch.zeros(shape, device=env.device, dtype=torch.bool)

        self._metric_count = torch.zeros(env.num_envs, device=env.device)
        self._last_metric_common_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        metric_names = (
            "left_virtual_force_error_n",
            "right_virtual_force_error_n",
            "mean_virtual_force_error_n",
            "virtual_force_clamp_fraction",
        )
        self._metric_sums = {
            name: torch.zeros(env.num_envs, device=env.device) for name in metric_names
        }
        self._latest_metrics = {
            name: torch.zeros(env.num_envs, device=env.device) for name in metric_names
        }
        self.refresh_reference_state()

    def _vector(
        self,
        value,
        expected_shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self._env.device, dtype=torch.float32)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}.")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return tensor

    @property
    def force_axis_mask_n(self) -> torch.Tensor:
        return self._force_axis_mask_n

    @property
    def position_axis_mask_n(self) -> torch.Tensor:
        return self._position_axis_mask_n

    @property
    def latest_force_tracking_errors(self) -> dict[str, torch.Tensor]:
        return self._latest_metrics

    def _spring_state_n(self) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        state_w = self._command_term.virtual_spring_state_world()
        q_wn = state_w["dataset_to_world_quaternion"]
        q_wn_hands = q_wn.unsqueeze(1).expand(-1, 2, -1)
        position_error_n = quat_apply_inverse(
            q_wn_hands.reshape(-1, 4),
            (state_w["target_position"] - state_w["actual_position"]).reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)
        velocity_error_n = quat_apply_inverse(
            q_wn_hands.reshape(-1, 4),
            (
                state_w["target_linear_velocity"] - state_w["actual_linear_velocity"]
            ).reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)
        return state_w, position_error_n, velocity_error_n

    def _target_force_robot(
        self,
        state_w: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_wn = state_w["dataset_to_world_quaternion"]
        q_wn_hands = q_wn.unsqueeze(1).expand(-1, 2, -1)
        csv_force_n = quat_apply_inverse(
            q_wn_hands.reshape(-1, 4),
            state_w["csv_hand_on_payload_force"].reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)
        sign = -1.0 if self.cfg.csv_force_is_hand_on_payload else 1.0
        target_force_n = sign * csv_force_n * self._force_axis_mask_n
        target_force_w = quat_apply(
            q_wn_hands.reshape(-1, 4),
            target_force_n.reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)
        return target_force_n, target_force_w

    def apply_physics_substep(self) -> None:
        """Recompute and buffer spring forces immediately before a sim step."""
        if bool(self.cfg.enabled) != self._enabled:
            raise RuntimeError(
                "Changing virtual_force.enabled at runtime is unsupported because "
                "Isaac Lab retains external-wrench buffers. Configure the mode "
                "before constructing the environment."
            )
        if not self._enabled:
            self.target_force_robot_w.zero_()
            self.actual_virtual_force_robot_w.zero_()
            self.force_error_robot_w.zero_()
            self.force_clamped.zero_()
            return

        state_w, position_error_n, velocity_error_n = self._spring_state_n()
        raw_force_n = self._kp * position_error_n + self._kd * velocity_error_n
        applied_force_n = torch.clamp(
            raw_force_n,
            min=-self._max_force,
            max=self._max_force,
        ) * self._force_axis_mask_n

        q_wn_hands = (
            state_w["dataset_to_world_quaternion"].unsqueeze(1).expand(-1, 2, -1)
        )
        applied_force_w = quat_apply(
            q_wn_hands.reshape(-1, 4),
            applied_force_n.reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)
        _, target_force_w = self._target_force_robot(state_w)

        # With is_global=True, Isaac Sim expects force, torque, and application
        # position in world coordinates. Applying at the measured palm anchor
        # avoids silently moving the wrench to the hand-link COM.
        self._robot.set_external_force_and_torque(
            forces=applied_force_w,
            torques=torch.zeros_like(applied_force_w),
            positions=state_w["actual_position"],
            body_ids=self._hand_body_ids,
            is_global=True,
        )

        self.target_force_robot_w.copy_(target_force_w)
        self.actual_virtual_force_robot_w.copy_(applied_force_w)
        self.spring_position_error_n.copy_(position_error_n)
        self.spring_velocity_error_n.copy_(velocity_error_n)
        self.force_error_robot_w.copy_(target_force_w - applied_force_w)
        self.force_clamped.copy_(
            (torch.abs(raw_force_n) > self._max_force) & (self._force_axis_mask_n > 0.0)
        )

    def refresh_reference_state(
        self,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Synchronize target/kinematic buffers without applying a force."""
        if env_ids is None:
            env_ids_tensor = torch.arange(
                self._env.num_envs, device=self._env.device, dtype=torch.long
            )
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self._env.device, dtype=torch.long
            )
        if len(env_ids_tensor) == 0:
            return

        state_w = self._command_term.virtual_spring_state_world(env_ids_tensor)
        q_wn_hands = state_w["dataset_to_world_quaternion"].unsqueeze(1).expand(-1, 2, -1)
        pos_error_n = quat_apply_inverse(
            q_wn_hands.reshape(-1, 4),
            (state_w["target_position"] - state_w["actual_position"]).reshape(-1, 3),
        ).reshape(len(env_ids_tensor), 2, 3)
        vel_error_n = quat_apply_inverse(
            q_wn_hands.reshape(-1, 4),
            (
                state_w["target_linear_velocity"] - state_w["actual_linear_velocity"]
            ).reshape(-1, 3),
        ).reshape(len(env_ids_tensor), 2, 3)

        csv_force_n = quat_apply_inverse(
            q_wn_hands.reshape(-1, 4),
            state_w["csv_hand_on_payload_force"].reshape(-1, 3),
        ).reshape(len(env_ids_tensor), 2, 3)
        sign = -1.0 if self.cfg.csv_force_is_hand_on_payload else 1.0
        target_force_n = sign * csv_force_n * self._force_axis_mask_n
        target_force_w = quat_apply(
            q_wn_hands.reshape(-1, 4),
            target_force_n.reshape(-1, 3),
        ).reshape(len(env_ids_tensor), 2, 3)

        self.target_force_robot_w[env_ids_tensor] = target_force_w
        self.actual_virtual_force_robot_w[env_ids_tensor] = 0.0
        self.spring_position_error_n[env_ids_tensor] = pos_error_n
        self.spring_velocity_error_n[env_ids_tensor] = vel_error_n
        self.force_error_robot_w[env_ids_tensor] = target_force_w
        self.force_clamped[env_ids_tensor] = False

    def current_kinematic_error_n(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current target-minus-actual anchor p/v errors in dataset axes."""
        _, position_error_n, velocity_error_n = self._spring_state_n()
        return position_error_n, velocity_error_n

    def _world_vectors_to_torso(self, vectors_w: torch.Tensor) -> torch.Tensor:
        torso_quat_w = self._robot.data.body_link_quat_w[:, self._torso_body_id]
        torso_quat_hands = torso_quat_w.unsqueeze(1).expand(-1, 2, -1)
        return quat_apply_inverse(
            torso_quat_hands.reshape(-1, 4),
            vectors_w.reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)

    def target_force_observation(self) -> torch.Tensor:
        """Target environment-on-hand forces in the current torso frame."""
        return self._world_vectors_to_torso(self.target_force_robot_w).reshape(
            self._env.num_envs, -1
        )

    def actual_force_observation(self) -> torch.Tensor:
        """Actual virtual spring forces in the current torso frame."""
        return self._world_vectors_to_torso(
            self.actual_virtual_force_robot_w
        ).reshape(self._env.num_envs, -1)

    def spring_deflection_observation(self) -> torch.Tensor:
        """Spring position error expressed in the current torso frame."""
        q_wn = self._command_term.virtual_spring_state_world()[
            "dataset_to_world_quaternion"
        ]
        q_wn_hands = q_wn.unsqueeze(1).expand(-1, 2, -1)
        error_w = quat_apply(
            q_wn_hands.reshape(-1, 4),
            self.spring_position_error_n.reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)
        return self._world_vectors_to_torso(error_w).reshape(self._env.num_envs, -1)

    def force_control_axis_observation(self) -> torch.Tensor:
        """Force-control axis directions expressed in the current torso frame."""
        q_wn = self._command_term.virtual_spring_state_world()[
            "dataset_to_world_quaternion"
        ]
        mask_n = self._force_axis_mask_n.view(1, 1, 3).expand(
            self._env.num_envs, 2, -1
        )
        q_wn_hands = q_wn.unsqueeze(1).expand(-1, 2, -1)
        mask_w = quat_apply(
            q_wn_hands.reshape(-1, 4),
            mask_n.reshape(-1, 3),
        ).reshape(self._env.num_envs, 2, 3)
        return self._world_vectors_to_torso(mask_w).reshape(self._env.num_envs, -1)

    def force_tracking_reward(self, std: float) -> torch.Tensor:
        """Paper-form exponential reward, averaged over the two hands."""
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError(f"Force reward std must be positive, got {std}.")
        if not self._enabled:
            return torch.zeros(self._env.num_envs, device=self._env.device)
        error = torch.norm(self.force_error_robot_w, dim=-1)
        return torch.exp(-error / std).mean(dim=1)

    def position_tracking_reward(self, std: float) -> torch.Tensor:
        """Position reward on axes complementary to force control."""
        position_error_n, _ = self.current_kinematic_error_n()
        error = torch.sum(
            torch.square(position_error_n * self._position_axis_mask_n), dim=-1
        )
        return torch.exp(-error / std**2).mean(dim=1)

    def linear_velocity_tracking_reward(self, std: float) -> torch.Tensor:
        """Linear-velocity reward on axes complementary to force control."""
        _, velocity_error_n = self.current_kinematic_error_n()
        error = torch.sum(
            torch.square(velocity_error_n * self._position_axis_mask_n), dim=-1
        )
        return torch.exp(-error / std**2).mean(dim=1)

    def record_metrics(self) -> None:
        """Accumulate force diagnostics once per policy step."""
        common_step = int(self._env.common_step_counter)
        env_ids = torch.nonzero(
            self._last_metric_common_step != common_step, as_tuple=False
        ).flatten()
        if len(env_ids) == 0:
            return

        force_error = torch.norm(self.force_error_robot_w[env_ids], dim=-1)
        values = {
            "left_virtual_force_error_n": force_error[:, 0],
            "right_virtual_force_error_n": force_error[:, 1],
            "mean_virtual_force_error_n": force_error.mean(dim=1),
            "virtual_force_clamp_fraction": (
                self.force_clamped[env_ids].float().sum(dim=(1, 2))
                / (2.0 * self._force_axis_mask_n.sum().clamp_min(1.0))
            ),
        }
        for name, value in values.items():
            self._latest_metrics[name][env_ids] = value
            self._metric_sums[name][env_ids] += value
        self._metric_count[env_ids] += 1.0
        self._last_metric_common_step[env_ids] = common_step

    def reset_episode_metrics(self, env_ids) -> dict[str, float]:
        """Return per-episode force logs and clear only the reset environments."""
        env_ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return {}
        count = self._metric_count[env_ids].clamp_min(1.0)
        logs = {
            f"Metrics/hand_reference/{name}": torch.mean(
                metric_sum[env_ids] / count
            ).item()
            for name, metric_sum in self._metric_sums.items()
        }
        self._metric_count[env_ids] = 0.0
        self._last_metric_common_step[env_ids] = -1
        for metric_sum in self._metric_sums.values():
            metric_sum[env_ids] = 0.0
        for latest in self._latest_metrics.values():
            latest[env_ids] = 0.0
        return logs


class CoopG1S2Env(ManagerBasedRLEnv):
    """S2 environment with explicit frozen-base/residual action bookkeeping."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Presence of this attribute makes train/play select the dict-action
        # wrapper that carries policy-side action decomposition into the env.
        self.action_extra_info: dict[str, torch.Tensor] = {}
        # ObservationManager probes term shapes inside the parent constructor,
        # before the command-backed virtual spring can be created.
        self._virtual_spring_initializing = True
        super().__init__(cfg, render_mode, **kwargs)
        self._residual_action_limit = float(cfg.residual_action_limit)
        if not math.isfinite(self._residual_action_limit) or self._residual_action_limit <= 0.0:
            raise ValueError("cfg.residual_action_limit must be finite and positive.")
        self._ensure_action_component_buffers()
        self._hand_reference_term = self.command_manager.get_term("hand_reference")
        self._virtual_spring = VirtualTwoHandForceManager(
            cfg.virtual_force,
            self,
            self._hand_reference_term,
        )
        self._virtual_spring_initializing = False
        # RSL-RL's wrapper calls reset before querying observations.  Alignment
        # belongs in _reset_idx, after reset events and the episode clock reset,
        # rather than here against the pre-reset spawn configuration.

    def _ensure_action_component_buffers(self) -> None:
        action_dim = self.action_manager.total_action_dim
        expected_shape = (self.num_envs, action_dim)
        for name in (
            "_s2_previous_base_action",
            "_s2_last_residual_action",
            "_s2_previous_residual_action",
        ):
            value = getattr(self, name, None)
            if value is None or value.shape != expected_shape:
                setattr(
                    self,
                    name,
                    torch.zeros(expected_shape, device=self.device, dtype=torch.float32),
                )
        scalar_specs = (
            "_s2_residual_metric_count",
            "_s2_residual_abs_sum",
            "_s2_residual_rms_sum",
            "_s2_residual_near_limit_sum",
            "_s2_residual_max_abs",
        )
        for name in scalar_specs:
            value = getattr(self, name, None)
            if value is None or value.shape != (self.num_envs,):
                setattr(
                    self,
                    name,
                    torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
                )

    def _record_residual_action_metrics(self, residual_action: torch.Tensor) -> None:
        """Accumulate arm-residual utilization without affecting the reward."""
        arm_residual = residual_action[:, S2_ARM_ACTION_START_INDEX:]
        abs_residual = torch.abs(arm_residual)
        self._s2_residual_metric_count += 1.0
        self._s2_residual_abs_sum += abs_residual.mean(dim=1)
        self._s2_residual_rms_sum += torch.sqrt(torch.mean(torch.square(arm_residual), dim=1))
        self._s2_residual_near_limit_sum += (
            abs_residual >= (0.95 * self._residual_action_limit)
        ).float().mean(dim=1)
        self._s2_residual_max_abs = torch.maximum(
            self._s2_residual_max_abs,
            abs_residual.max(dim=1).values,
        )

    def _reset_residual_action_metrics(self, env_ids) -> dict[str, float]:
        """Return per-episode residual utilization and clear reset environments."""
        if not hasattr(self, "_s2_residual_metric_count"):
            return {}
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return {}
        count = self._s2_residual_metric_count[env_ids].clamp_min(1.0)
        logs = {
            "Metrics/residual/mean_arm_abs": torch.mean(
                self._s2_residual_abs_sum[env_ids] / count
            ).item(),
            "Metrics/residual/mean_arm_rms": torch.mean(
                self._s2_residual_rms_sum[env_ids] / count
            ).item(),
            "Metrics/residual/near_limit_fraction": torch.mean(
                self._s2_residual_near_limit_sum[env_ids] / count
            ).item(),
            "Metrics/residual/episode_max_abs": torch.mean(
                self._s2_residual_max_abs[env_ids]
            ).item(),
        }
        self._s2_residual_metric_count[env_ids] = 0.0
        self._s2_residual_abs_sum[env_ids] = 0.0
        self._s2_residual_rms_sum[env_ids] = 0.0
        self._s2_residual_near_limit_sum[env_ids] = 0.0
        self._s2_residual_max_abs[env_ids] = 0.0
        return logs

    def step(self, action_dict):
        """Execute total action while retaining its frozen-base decomposition.

        PPO returns the physical action in ``action`` and detached diagnostic
        components in ``extra_info``.  Requiring this contract avoids silently
        feeding total actions into the frozen S1 history after a wrapper or
        policy configuration mistake.
        """
        if not isinstance(action_dict, dict):
            raise TypeError(
                "CoopG1S2Env requires {'action': total_action, 'extra_info': ...}; "
                "use RslRlVecEnvWrapperDictAction."
            )
        if "action" not in action_dict or "extra_info" not in action_dict:
            raise KeyError("S2 action dict must contain both 'action' and 'extra_info'.")

        total_action = action_dict["action"]
        extra_info = action_dict["extra_info"]
        if not isinstance(extra_info, dict) or "base_action" not in extra_info:
            raise KeyError(
                "ActorCriticResidual must provide extra_info['base_action']; "
                "otherwise S1 action history cannot be reconstructed."
            )

        self._ensure_action_component_buffers()
        base_action = extra_info["base_action"].to(
            device=total_action.device,
            dtype=total_action.dtype,
        )
        if total_action.shape != self._s2_last_residual_action.shape:
            raise ValueError(
                f"Total action shape {tuple(total_action.shape)} does not match "
                f"S2 action buffers {tuple(self._s2_last_residual_action.shape)}."
            )
        if base_action.shape != total_action.shape:
            raise ValueError(
                f"Base action shape {tuple(base_action.shape)} does not match "
                f"total action shape {tuple(total_action.shape)}."
            )

        self._s2_previous_residual_action.copy_(self._s2_last_residual_action)
        self._s2_previous_base_action.copy_(base_action.detach())
        residual_action = (total_action - base_action).detach()
        self._s2_last_residual_action.copy_(residual_action)
        self._record_residual_action_metrics(residual_action)
        self.action_extra_info = extra_info

        # Rewards are evaluated after physics and before CommandManager.compute.
        # Advancing here makes the post-physics state compare against t + dt.
        self._hand_reference_term.update_from_episode_step(preview_steps=1)
        return super().step(total_action)

    def _reset_idx(self, env_ids):
        residual_logs = self._reset_residual_action_metrics(env_ids)
        tracking_logs = {}
        if hasattr(self, "_hand_reference_term"):
            tracking_logs = self._hand_reference_term.reset_episode_metrics(env_ids)
        force_logs = {}
        if hasattr(self, "_virtual_spring"):
            force_logs = self._virtual_spring.reset_episode_metrics(env_ids)
        super()._reset_idx(env_ids)
        if residual_logs:
            self.extras["log"].update(residual_logs)
        if tracking_logs:
            self.extras["log"].update(tracking_logs)
        if force_logs:
            self.extras["log"].update(force_logs)
        if hasattr(self, "_s2_previous_base_action"):
            self._s2_previous_base_action[env_ids] = 0.0
            self._s2_last_residual_action[env_ids] = 0.0
            self._s2_previous_residual_action[env_ids] = 0.0
        if hasattr(self, "_hand_reference_term"):
            # Reset events write the new root/joint state directly to PhysX and
            # invalidate ArticulationData's link-pose timestamps.  Reading the
            # hand pose inside align_to_current_state therefore triggers
            # update_articulations_kinematic() for the reset configuration.
            # Do not add a physics step (or a second scene reset) here.
            self._hand_reference_term.align_to_current_state(env_ids)
        if hasattr(self, "_virtual_spring"):
            self._virtual_spring.refresh_reference_state(env_ids)

    def refresh_hand_reference_alignment(
        self,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Realign CSV time zero to the current simulated grasp."""
        self._hand_reference_term.align_to_current_state(env_ids)
