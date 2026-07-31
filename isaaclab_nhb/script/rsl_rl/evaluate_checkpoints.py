# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate RSL-RL checkpoints with fixed velocity commands.

This script is intentionally separate from play.py: play.py is for visual
inspection, while this script produces report-ready CSV metrics.

Example:
    python script/rsl_rl/evaluate_checkpoints.py \
        --task CoopG1S1-29dof-HoldBox \
        --checkpoint logs/rsl_rl/coopG1S1/.../model_12998.pt \
        --policy_name s1_0622 \
        --velocities 0.25 0.30 0.35 0.40 \
        --num_envs 64 \
        --max_episodes 100 \
        --device cuda:5 \
        --headless
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import isaaclab_nhb  # noqa: F401
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate RSL-RL checkpoints with fixed commands.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument(
    "--checkpoint",
    action="append",
    required=True,
    help="Checkpoint path. Can be passed multiple times.",
)
parser.add_argument(
    "--policy_name",
    action="extend",
    nargs="+",
    default=None,
    help="Display name for each checkpoint. Can be passed multiple times.",
)
parser.add_argument("--velocities", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40])
parser.add_argument("--lin_vel_y", type=float, default=0.0)
parser.add_argument("--ang_vel_z", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_episodes", type=int, default=100)
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=None, help="Optional hard cap per velocity evaluation.")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--save_trajectory", action="store_true", help="Save a single env-0 trajectory per run.")
parser.add_argument("--trajectory_env_index", type=int, default=0)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--no_onnx_export", action="store_true", help="Kept for CLI symmetry; no ONNX is exported here.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.headless:
    isaaclab_nhb.HEADLESS_FLAG = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils.vecenv_wrapper import RslRlVecEnvWrapperDictAction

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_nhb import *  # noqa: F401,F403
from isaaclab_nhb.tasks.humanoid.coopG1S1.coopG1S1_env_cfg import HOLD_HAND_TARGET_POS


class RslRlVecEnvWrapperExtraInfo(RslRlVecEnvWrapper):
    """Accept local rsl_rl policies that return (actions, extra_info)."""

    def step(self, actions: torch.Tensor, extra_info=None):
        return super().step(actions)


@dataclass
class RunningStats:
    sum: float = 0.0
    sum_sq: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach()
        self.sum += float(values.sum().cpu())
        self.sum_sq += float((values * values).sum().cpu())
        self.count += int(values.numel())

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else float("nan")

    @property
    def rmse(self) -> float:
        return (self.sum_sq / self.count) ** 0.5 if self.count > 0 else float("nan")

    @property
    def std(self) -> float:
        if self.count <= 1:
            return float("nan")
        mean = self.mean
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return var**0.5


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir(task_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return _project_root() / "report_materials" / "evaluations" / f"{timestamp}_{task_name}"


def _resolve_checkpoint(path: str) -> str:
    path = os.path.expanduser(path.strip().strip("'\""))
    if not os.path.isabs(path):
        project_path = _project_root() / path
        if project_path.is_file():
            path = str(project_path)
    path = retrieve_file_path(path)
    if not path.endswith(".pt"):
        raise ValueError(f"Checkpoint must be a .pt file: {path}")
    return path


def _policy_names(checkpoints: list[str]) -> list[str]:
    if args_cli.policy_name is None:
        return [Path(path).parent.name for path in checkpoints]
    if len(args_cli.policy_name) != len(checkpoints):
        raise ValueError("--policy_name must be passed once per --checkpoint.")
    return args_cli.policy_name


def _set_fixed_command(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, vx: float) -> None:
    command = env_cfg.commands.base_velocity
    command.ranges.lin_vel_x = (vx, vx)
    command.ranges.lin_vel_y = (args_cli.lin_vel_y, args_cli.lin_vel_y)
    command.ranges.ang_vel_z = (args_cli.ang_vel_z, args_cli.ang_vel_z)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    if hasattr(command, "rel_heading_envs"):
        command.rel_heading_envs = 0.0
    if hasattr(command, "debug_vis"):
        command.debug_vis = False


def _disable_reset_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if hasattr(env_cfg.events, "reset_base"):
        reset_base = env_cfg.events.reset_base
        reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        if "velocity_range" in reset_base.params:
            for key in reset_base.params["velocity_range"]:
                reset_base.params["velocity_range"][key] = (0.0, 0.0)
    if hasattr(env_cfg.events, "reset_robot_joints"):
        reset_joints = env_cfg.events.reset_robot_joints
        reset_joints.params["position_range"] = (1.0, 1.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)


def _wrap_env(env, agent_cfg: RslRlBaseRunnerCfg):
    wrapper_cls = RslRlVecEnvWrapperDictAction if hasattr(env.unwrapped, "action_extra_info") else RslRlVecEnvWrapperExtraInfo
    return wrapper_cls(env, clip_actions=agent_cfg.clip_actions)


def _load_policy(env, agent_cfg: RslRlBaseRunnerCfg, checkpoint: str):
    if getattr(agent_cfg.policy, "class_name", None) == "ActorCriticResidual":
        # S2 checkpoints contain their frozen S1 actor and normalizers.  Do
        # not require the original S1 checkpoint merely to construct the
        # policy used for evaluation.
        agent_cfg.policy.defer_base_policy_load = True
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint, load_optimizer=False)
    return runner.get_inference_policy(device=env.unwrapped.device)


def _body_ids(robot) -> dict[str, int | list[int]]:
    ids = {
        "torso": robot.body_names.index("torso_link"),
        "pelvis": robot.body_names.index("pelvis"),
    }
    hand_names = ["left_rubber_hand", "right_rubber_hand"]
    if all(name in robot.body_names for name in hand_names):
        ids["hands"] = [robot.body_names.index(name) for name in hand_names]
    return ids


def _hand_position_error(robot, ids: dict[str, int | list[int]]) -> torch.Tensor | None:
    if "hands" not in ids:
        return None
    torso_id = ids["torso"]
    hand_ids = ids["hands"]
    torso_pos_w = robot.data.body_link_pos_w[:, torso_id, :]
    torso_quat_w = robot.data.body_link_quat_w[:, torso_id, :]
    hand_pos_w = robot.data.body_link_pos_w[:, hand_ids, :]
    hand_rel_w = hand_pos_w - torso_pos_w.unsqueeze(1)
    hand_rel_b = quat_apply_inverse(
        torso_quat_w.unsqueeze(1).expand(-1, len(hand_ids), -1).reshape(-1, 4),
        hand_rel_w.reshape(-1, 3),
    ).reshape(hand_rel_w.shape[0], len(hand_ids), 3)
    target = torch.tensor(HOLD_HAND_TARGET_POS, dtype=hand_rel_b.dtype, device=hand_rel_b.device)
    return torch.linalg.norm(hand_rel_b - target.unsqueeze(0), dim=-1).mean(dim=-1)


def _current_metrics(env, previous_actions: torch.Tensor | None, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    ids = _body_ids(robot)
    command = env.unwrapped.command_manager.get_command("base_velocity")

    base_vel_b = robot.data.root_lin_vel_b
    speed_error_xy = torch.linalg.norm(command[:, :2] - base_vel_b[:, :2], dim=-1)
    vx_error = base_vel_b[:, 0] - command[:, 0]
    vy_error = base_vel_b[:, 1] - command[:, 1]
    torso_ang_vel_xy = torch.linalg.norm(robot.data.body_link_ang_vel_w[:, ids["torso"], :2], dim=-1)
    pelvis_ang_vel_xy = torch.linalg.norm(robot.data.body_link_ang_vel_w[:, ids["pelvis"], :2], dim=-1)

    if previous_actions is None:
        action_rate = torch.zeros(actions.shape[0], device=actions.device)
    else:
        action_rate = torch.linalg.norm(actions - previous_actions, dim=-1)

    metrics = {
        "command_vx": command[:, 0],
        "actual_vx": base_vel_b[:, 0],
        "actual_vy": base_vel_b[:, 1],
        "speed_error_xy": speed_error_xy,
        "vx_error": vx_error,
        "vy_error": vy_error,
        "torso_ang_vel_xy": torso_ang_vel_xy,
        "pelvis_ang_vel_xy": pelvis_ang_vel_xy,
        "action_rate": action_rate,
    }
    hand_error = _hand_position_error(robot, ids)
    if hand_error is not None:
        metrics["hand_pos_error"] = hand_error
    return metrics


def _write_trajectory_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "step",
            "time_s",
            "command_vx",
            "actual_vx",
            "actual_vy",
            "speed_error_xy",
            "vx_error",
            "vy_error",
            "hand_pos_error",
            "torso_ang_vel_xy",
            "pelvis_ang_vel_xy",
            "action_rate",
            "done",
            "timeout",
            "terminated",
        ]
    )


def _trajectory_row(
    step: int,
    dt: float,
    metrics: dict[str, torch.Tensor],
    env_index: int,
    dones: torch.Tensor,
    time_outs: torch.Tensor,
    terminated: torch.Tensor,
) -> list[float | int | str]:
    def value(name: str) -> float | str:
        if name not in metrics:
            return ""
        return float(metrics[name][env_index].detach().cpu())

    return [
        step,
        step * dt,
        value("command_vx"),
        value("actual_vx"),
        value("actual_vy"),
        value("speed_error_xy"),
        value("vx_error"),
        value("vy_error"),
        value("hand_pos_error"),
        value("torso_ang_vel_xy"),
        value("pelvis_ang_vel_xy"),
        value("action_rate"),
        int(bool(dones[env_index].detach().cpu())),
        int(bool(time_outs[env_index].detach().cpu())),
        int(bool(terminated[env_index].detach().cpu())),
    ]


def _evaluate_one(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint: str,
    policy_name: str,
    vx: float,
    output_dir: Path,
) -> dict[str, float | int | str]:
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg, "rebuild_dynamic_cfg"):
        env_cfg.rebuild_dynamic_cfg()
    _set_fixed_command(env_cfg, vx)
    _disable_reset_randomization(env_cfg)
    env_cfg.curriculum = None
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = str(Path(checkpoint).parent)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = _wrap_env(env, agent_cfg)

    policy = _load_policy(env, agent_cfg, checkpoint)
    obs = env.get_observations()
    dt = float(env.unwrapped.step_dt)
    max_episode_length = int(env.unwrapped.max_episode_length)
    max_steps = args_cli.max_steps or int(max_episode_length * max(2, (args_cli.max_episodes // args_cli.num_envs) + 2))

    stats = {
        "speed_error_xy": RunningStats(),
        "vx_error": RunningStats(),
        "vy_error": RunningStats(),
        "hand_pos_error": RunningStats(),
        "torso_ang_vel_xy": RunningStats(),
        "pelvis_ang_vel_xy": RunningStats(),
        "action_rate": RunningStats(),
    }
    completed_episodes = 0
    timeout_episodes = 0
    terminated_episodes = 0
    previous_actions = None

    trajectory_file = None
    trajectory_writer = None
    trajectory_path = ""
    if args_cli.save_trajectory:
        safe_policy = policy_name.replace("/", "_").replace(" ", "_")
        trajectory_dir = output_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = str(trajectory_dir / f"{safe_policy}_vx{vx:.2f}.csv")
        trajectory_file = open(trajectory_path, "w", newline="")
        trajectory_writer = csv.writer(trajectory_file)
        _write_trajectory_header(trajectory_writer)

    try:
        for step in range(max_steps):
            with torch.inference_mode():
                actions, extra_info = policy(obs)
                obs, _, dones, extras = env.step(actions, extra_info)

                if step >= args_cli.warmup_steps:
                    metrics = _current_metrics(env, previous_actions, actions)
                    active = dones == 0
                    for name, stat in stats.items():
                        if name in metrics:
                            stat.update(metrics[name][active])

                    time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)).bool()
                    terminated = dones.bool() & ~time_outs

                    if trajectory_writer is not None:
                        env_index = min(args_cli.trajectory_env_index, env.num_envs - 1)
                        trajectory_writer.writerow(
                            _trajectory_row(step, dt, metrics, env_index, dones, time_outs, terminated)
                        )

                    if dones.any():
                        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                        completed_episodes += int(done_ids.numel())
                        timeout_episodes += int(time_outs[done_ids].sum().item())
                        terminated_episodes += int(terminated[done_ids].sum().item())

                        if completed_episodes >= args_cli.max_episodes:
                            break

                previous_actions = actions.detach().clone()
    finally:
        if trajectory_file is not None:
            trajectory_file.close()
        env.close()

    if completed_episodes == 0:
        success_rate = 0.0
        failure_rate = 0.0
    else:
        success_rate = timeout_episodes / completed_episodes
        failure_rate = terminated_episodes / completed_episodes

    return {
        "policy_name": policy_name,
        "task": args_cli.task,
        "checkpoint": checkpoint,
        "v_cmd_x": vx,
        "num_envs": args_cli.num_envs,
        "requested_episodes": args_cli.max_episodes,
        "completed_episodes": completed_episodes,
        "timeout_episodes": timeout_episodes,
        "terminated_episodes": terminated_episodes,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "speed_rmse_xy": stats["speed_error_xy"].rmse,
        "speed_error_mean_xy": stats["speed_error_xy"].mean,
        "speed_error_std_xy": stats["speed_error_xy"].std,
        "vx_error_mean": stats["vx_error"].mean,
        "vx_error_rmse": stats["vx_error"].rmse,
        "vy_error_mean": stats["vy_error"].mean,
        "hand_pos_error_mean": stats["hand_pos_error"].mean,
        "torso_ang_vel_xy_mean": stats["torso_ang_vel_xy"].mean,
        "pelvis_ang_vel_xy_mean": stats["pelvis_ang_vel_xy"].mean,
        "action_rate_mean": stats["action_rate"].mean,
        "max_episode_length_steps": max_episode_length,
        "trajectory_csv": trajectory_path,
    }


def _write_summary_header(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    checkpoints = [_resolve_checkpoint(path) for path in args_cli.checkpoint]
    policy_names = _policy_names(checkpoints)
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else _default_output_dir(args_cli.task)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for checkpoint, policy_name in zip(checkpoints, policy_names):
        for vx in args_cli.velocities:
            print(f"[EVAL] policy={policy_name} vx={vx:.2f} checkpoint={checkpoint}")
            row = _evaluate_one(env_cfg, agent_cfg, checkpoint, policy_name, vx, output_dir)
            rows.append(row)
            summary_path = output_dir / "eval_summary.csv"
            _write_summary_header(summary_path, rows)
            print(
                "[EVAL] "
                f"success={row['success_rate']:.3f} "
                f"speed_rmse={row['speed_rmse_xy']:.4f} "
                f"hand_error={row['hand_pos_error_mean']:.4f} "
                f"episodes={row['completed_episodes']}"
            )

    print(f"[INFO] Evaluation summary saved to: {output_dir / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
