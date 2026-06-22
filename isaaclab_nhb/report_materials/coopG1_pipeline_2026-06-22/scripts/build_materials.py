#!/usr/bin/env python3
"""Build a compact, reproducible report-material bundle from CoopG1 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_PROJECT_ROOT = Path("/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb")
DEFAULT_OUTPUT = DEFAULT_PROJECT_ROOT / "report_materials" / "coopG1_pipeline_2026-06-22"
EXPERIMENTS = ("coopG1S0", "coopG1S1", "coopG1S2")
SELECTED_SCALARS = (
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Episode_Reward/track_lin_vel_xy",
    "Episode_Reward/track_lin_vel_xy_fine",
    "Episode_Reward/hand_payload_pose",
    "Episode_Reward/gait",
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_noise_std",
    "Loss/learning_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=yaml.UnsafeLoader)
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {"_parse_error": str(exc)}


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def event_scalars(run_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    merged: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for event_file in sorted(run_dir.glob("events.out.tfevents.*")):
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    previous = merged[tag].get(event.step)
                    if previous is None or event.wall_time >= previous[0]:
                        merged[tag][event.step] = (event.wall_time, float(event.value))
        except Exception as exc:
            merged["_event_error"][-1] = (0.0, float("nan"))
            print(f"[WARN] Failed to parse {event_file}: {exc}")
    return {
        tag: [(step, wall_value[0], wall_value[1]) for step, wall_value in sorted(by_step.items())]
        for tag, by_step in merged.items()
    }


def downsample(points: list[tuple[int, float, float]], maximum: int = 1000) -> list[tuple[int, float, float]]:
    if len(points) <= maximum:
        return points
    stride = math.ceil(len(points) / maximum)
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def reward_cfg(env_cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = nested(env_cfg, "rewards", name, default={})
    return value if isinstance(value, dict) else {}


def reward_weight(env_cfg: dict[str, Any], name: str) -> Any:
    return reward_cfg(env_cfg, name).get("weight")


def reward_param(env_cfg: dict[str, Any], name: str, param: str) -> Any:
    return nested(reward_cfg(env_cfg, name), "params", param)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_hydra_outputs(project_root: Path, output: Path) -> list[dict[str, str]]:
    """Preserve compact Hydra snapshots, including launches that produced no training run."""
    source_root = project_root / "outputs"
    copied_root = output / "hydra_outputs"
    rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, str]] = []
    for config_path in sorted(source_root.glob("*/*/.hydra/config.yaml")):
        run_dir = config_path.parents[1]
        relative_run = run_dir.relative_to(source_root)
        destination_dir = copied_root / relative_run
        for name in ("config.yaml", "overrides.yaml"):
            source = run_dir / ".hydra" / name
            destination = destination_dir / name
            if copy_if_exists(source, destination):
                external_rows.append(
                    {
                        "copy": str(destination),
                        "category": "hydra_output",
                        "source": str(source),
                    }
                )
        config = safe_yaml(config_path)
        env_cfg = config.get("env", {}) if isinstance(config.get("env"), dict) else {}
        agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
        policy_cfg = agent_cfg.get("policy", {}) if isinstance(agent_cfg.get("policy"), dict) else {}
        rows.append(
            {
                "date": relative_run.parts[0],
                "time": relative_run.parts[1],
                "source_path": str(run_dir),
                "num_envs": nested(env_cfg, "scene", "num_envs"),
                "sim_device": nested(env_cfg, "sim", "device"),
                "seed": agent_cfg.get("seed"),
                "max_iterations": agent_cfg.get("max_iterations"),
                "experiment_name": agent_cfg.get("experiment_name"),
                "run_name": agent_cfg.get("run_name"),
                "resume": agent_cfg.get("resume"),
                "load_run": agent_cfg.get("load_run"),
                "load_checkpoint": agent_cfg.get("load_checkpoint"),
                "policy_class": policy_cfg.get("class_name"),
            }
        )
    write_csv(output / "metrics" / "hydra_output_index.csv", rows)
    return external_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(log_root: Path, output: Path):
    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    final_scalar_rows: list[dict[str, Any]] = []
    series_by_run: dict[tuple[str, str], dict[str, list[tuple[int, float, float]]]] = {}

    for experiment in EXPERIMENTS:
        experiment_dir = log_root / experiment
        if not experiment_dir.is_dir():
            continue
        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            env_path = run_dir / "params" / "env.yaml"
            agent_path = run_dir / "params" / "agent.yaml"
            env_cfg = safe_yaml(env_path)
            agent_cfg = safe_yaml(agent_path)
            scalars = event_scalars(run_dir)
            series_by_run[(experiment, run_dir.name)] = scalars

            copied_config_dir = output / "configs" / experiment / run_dir.name
            copy_if_exists(env_path, copied_config_dir / "env.yaml")
            copy_if_exists(agent_path, copied_config_dir / "agent.yaml")

            checkpoints = sorted(run_dir.glob("model_*.pt"), key=checkpoint_step)
            for checkpoint in checkpoints:
                checkpoint_rows.append(
                    {
                        "experiment": experiment,
                        "run": run_dir.name,
                        "step": checkpoint_step(checkpoint),
                        "size_bytes": checkpoint.stat().st_size,
                        "path": str(checkpoint.resolve()),
                    }
                )

            policy_cfg = nested(env_cfg, "observations", "policy", default={})
            policy_terms = [
                key
                for key, value in policy_cfg.items()
                if isinstance(value, dict) and isinstance(value.get("func"), str)
            ] if isinstance(policy_cfg, dict) else []
            commands = nested(env_cfg, "commands", "base_velocity", "ranges", default={})
            commands = commands if isinstance(commands, dict) else {}

            summary: dict[str, Any] = {
                "experiment": experiment,
                "run": run_dir.name,
                "run_path": str(run_dir.resolve()),
                "has_env_yaml": env_path.is_file(),
                "has_agent_yaml": agent_path.is_file(),
                "event_files": len(list(run_dir.glob("events.out.tfevents.*"))),
                "checkpoint_count": len(checkpoints),
                "latest_checkpoint_step": checkpoint_step(checkpoints[-1]) if checkpoints else None,
                "latest_checkpoint_path": str(checkpoints[-1].resolve()) if checkpoints else "",
                "checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
                "num_envs": nested(env_cfg, "scene", "num_envs"),
                "seed": agent_cfg.get("seed"),
                "max_iterations_cfg": agent_cfg.get("max_iterations"),
                "resume": agent_cfg.get("resume"),
                "load_run": agent_cfg.get("load_run"),
                "load_checkpoint": agent_cfg.get("load_checkpoint"),
                "actor_class": nested(agent_cfg, "policy", "class_name"),
                "actor_hidden_dims": json_text(nested(agent_cfg, "policy", "actor_hidden_dims")),
                "critic_hidden_dims": json_text(nested(agent_cfg, "policy", "critic_hidden_dims")),
                "policy_history_length": policy_cfg.get("history_length") if isinstance(policy_cfg, dict) else None,
                "policy_terms": "|".join(policy_terms),
                "policy_has_base_lin_vel": "base_lin_vel" in policy_terms,
                "policy_has_base_ang_vel": "base_ang_vel" in policy_terms,
                "command_lin_vel_x": json_text(commands.get("lin_vel_x")),
                "command_lin_vel_y": json_text(commands.get("lin_vel_y")),
                "command_ang_vel_z": json_text(commands.get("ang_vel_z")),
                "track_vel_weight": reward_weight(env_cfg, "track_lin_vel_xy"),
                "track_vel_std": reward_param(env_cfg, "track_lin_vel_xy", "std"),
                "track_vel_fine_weight": reward_weight(env_cfg, "track_lin_vel_xy_fine"),
                "track_vel_fine_std": reward_param(env_cfg, "track_lin_vel_xy_fine", "std"),
                "hand_pose_weight": reward_weight(env_cfg, "hand_payload_pose"),
                "arm_pose_weight": reward_weight(env_cfg, "arm_target_pose"),
                "gait_weight": reward_weight(env_cfg, "gait"),
                "feet_air_time_weight": reward_weight(env_cfg, "feet_air_time"),
                "torso_ang_vel_weight": reward_weight(env_cfg, "torso_ang_vel"),
                "upright_torso_weight": reward_weight(env_cfg, "upright_torso"),
                "pelvis_ang_vel_weight": reward_weight(env_cfg, "pelvis_ang_vel"),
                "upright_pelvis_weight": reward_weight(env_cfg, "upright_pelvis"),
            }
            actual_last_steps = [points[-1][0] for tag, points in scalars.items() if points and tag != "_event_error"]
            summary["actual_last_iteration"] = max(actual_last_steps) if actual_last_steps else None
            for tag in SELECTED_SCALARS:
                points = scalars.get(tag, [])
                summary["final_" + tag.replace("/", "__")] = points[-1][2] if points else None
            run_rows.append(summary)

            rewards = env_cfg.get("rewards", {})
            if isinstance(rewards, dict):
                for term, cfg in rewards.items():
                    if not isinstance(cfg, dict):
                        continue
                    reward_rows.append(
                        {
                            "experiment": experiment,
                            "run": run_dir.name,
                            "term": term,
                            "weight": cfg.get("weight"),
                            "func": cfg.get("func"),
                            "params": json_text(cfg.get("params")),
                        }
                    )

            observations = env_cfg.get("observations", {})
            if isinstance(observations, dict):
                for group_name, group_cfg in observations.items():
                    if not isinstance(group_cfg, dict):
                        continue
                    for term, cfg in group_cfg.items():
                        if not isinstance(cfg, dict) or not isinstance(cfg.get("func"), str):
                            continue
                        observation_rows.append(
                            {
                                "experiment": experiment,
                                "run": run_dir.name,
                                "group": group_name,
                                "term": term,
                                "func": cfg.get("func"),
                                "noise": json_text(cfg.get("noise")),
                                "scale": json_text(cfg.get("scale")),
                                "clip": json_text(cfg.get("clip")),
                            }
                        )

            for tag, points in scalars.items():
                if not points or tag == "_event_error":
                    continue
                final_scalar_rows.append(
                    {
                        "experiment": experiment,
                        "run": run_dir.name,
                        "tag": tag,
                        "final_step": points[-1][0],
                        "final_value": points[-1][2],
                    }
                )

            timeseries_rows: list[dict[str, Any]] = []
            for tag in SELECTED_SCALARS:
                for step, wall_time, value in downsample(scalars.get(tag, [])):
                    timeseries_rows.append(
                        {"step": step, "wall_time": wall_time, "tag": tag, "value": value}
                    )
            if timeseries_rows:
                write_csv(output / "metrics" / "timeseries" / experiment / f"{run_dir.name}.csv", timeseries_rows)

    run_rows.sort(key=lambda row: (row["experiment"], row["run"]))
    write_csv(output / "metrics" / "run_summary.csv", run_rows)
    write_csv(output / "metrics" / "checkpoint_index.csv", checkpoint_rows)
    write_csv(output / "metrics" / "reward_terms.csv", reward_rows)
    write_csv(output / "metrics" / "observation_terms.csv", observation_rows)
    write_csv(output / "metrics" / "final_scalar_metrics.csv", final_scalar_rows)
    return run_rows, checkpoint_rows, final_scalar_rows, series_by_run


def short_label(run_name: str) -> str:
    label = re.sub(r"^2026-", "", run_name)
    label = label.replace("_s1_", " ").replace("_16k_", " ")
    return label[:48]


def plot_velocity_summary(run_rows: list[dict[str, Any]], output: Path) -> None:
    rows = [
        row for row in run_rows
        if row["experiment"] == "coopG1S1" and row.get("final_Metrics__base_velocity__error_vel_xy") is not None
    ]
    if not rows:
        return
    values = [float(row["final_Metrics__base_velocity__error_vel_xy"]) for row in rows]
    colors = ["#2f6fa3" if not row.get("policy_has_base_lin_vel") else "#d9782d" for row in rows]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_ylabel("Final XY velocity tracking error (m/s)")
    ax.set_title("S1 tuning runs: final velocity tracking error")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([short_label(row["run"]) for row in rows], rotation=70, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2f6fa3", label="without policy base_lin_vel"),
            plt.Rectangle((0, 0), 1, 1, color="#d9782d", label="with policy base_lin_vel"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    path = output / "figures" / "training" / "s1_final_velocity_error_all_runs.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def selected_run_keys(series_by_run):
    patterns = (
        "scratch_forward_footdist_airtime",
        "scratch_gait_pelvis_feetorient_holdbox",
        "hand8_torso_pelvis_rp_stronger",
        "base_velocity_scratch_16k_10k",
        "base_velocity_precise",
    )
    selected = []
    for key in series_by_run:
        if key[0] != "coopG1S1":
            continue
        if any(pattern in key[1] for pattern in patterns):
            selected.append(key)
    return selected


def plot_selected_curves(series_by_run, output: Path) -> None:
    selected = selected_run_keys(series_by_run)
    if not selected:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)
    for key in selected:
        scalars = series_by_run[key]
        label = short_label(key[1])
        for ax, tag in zip(
            axes,
            ("Metrics/base_velocity/error_vel_xy", "Episode_Reward/track_lin_vel_xy"),
            strict=True,
        ):
            points = downsample(scalars.get(tag, []), maximum=1500)
            if points:
                ax.plot([point[0] for point in points], [point[2] for point in points], label=label, linewidth=1.4)
    axes[0].set_title("Selected S1 runs: XY velocity error")
    axes[0].set_ylabel("Error (m/s)")
    axes[1].set_title("Selected S1 runs: broad velocity reward")
    axes[1].set_ylabel("Weighted reward")
    axes[1].set_xlabel("Training iteration")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = output / "figures" / "training" / "s1_selected_training_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_latest_dashboard(series_by_run, output: Path) -> None:
    candidates = sorted(
        key
        for key, scalars in series_by_run.items()
        if key[0] == "coopG1S1" and scalars.get("Train/mean_reward")
    )
    if not candidates:
        return
    key = candidates[-1]
    scalars = series_by_run[key]
    tags = (
        "Metrics/base_velocity/error_vel_xy",
        "Episode_Reward/track_lin_vel_xy",
        "Train/mean_reward",
        "Policy/mean_noise_std",
    )
    titles = ("XY velocity error", "Velocity reward", "Mean reward", "Policy noise std")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, tag, title in zip(axes.flat, tags, titles, strict=True):
        points = downsample(scalars.get(tag, []), maximum=1500)
        if points:
            ax.plot([point[0] for point in points], [point[2] for point in points], color="#2f6fa3")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Latest S1 run dashboard: {key[1]}")
    fig.tight_layout()
    path = output / "figures" / "training" / "latest_s1_dashboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_reward_contributions(final_scalar_rows: list[dict[str, Any]], output: Path) -> None:
    preferred = "2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k"
    rows = [
        row for row in final_scalar_rows
        if row["experiment"] == "coopG1S1"
        and row["run"] == preferred
        and row["tag"].startswith("Episode_Reward/")
    ]
    if not rows:
        return
    rows.sort(key=lambda row: abs(float(row["final_value"])), reverse=True)
    rows = rows[:18]
    rows.reverse()
    values = [float(row["final_value"]) for row in rows]
    labels = [row["tag"].split("/", 1)[1] for row in rows]
    colors = ["#3b8c6e" if value >= 0 else "#b95c59" for value in values]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Final weighted episode reward")
    ax.set_title("Reward contributions: S1 base-velocity run at iteration 9999")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output / "figures" / "training" / "s1_reward_contributions_iter9999.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def copy_sources(project_root: Path, output: Path) -> list[dict[str, str]]:
    home = Path("/home/zhaowenhao")
    sources = [
        project_root / "tasks/humanoid/coopG1S0/coopG1S0_env_cfg.py",
        project_root / "tasks/humanoid/coopG1S1/coopG1S1_env_cfg.py",
        project_root / "tasks/humanoid/coopG1S1/agents/coopG1S1_rsl_rl_ppo_cfg.py",
        project_root / "tasks/humanoid/coopG1S2/coopG1S2_env_cfg.py",
        project_root / "tasks/humanoid/coopG1S2/coopG1S2_env.py",
        project_root / "tasks/humanoid/coopG1S2/agents/coopG1S2_rsl_rl_ppo_cfg.py",
        project_root / "tasks/mdp_nhb/commands.py",
        project_root / "tasks/mdp_nhb/commands_cfg.py",
        project_root / "tasks/mdp_nhb/observations.py",
        project_root / "tasks/mdp_nhb/rewards.py",
        project_root / "tasks/rl_cfg/rl_cfg.py",
        project_root / "script/rsl_rl/train.py",
        project_root / "script/rsl_rl/play.py",
        home / "rsl_rl/rsl_rl/modules/actor_critic.py",
        home / "rsl_rl/rsl_rl/modules/actor_critic_residual.py",
        home / "rsl_rl/rsl_rl/modules/__init__.py",
        home / "rsl_rl/rsl_rl/runners/on_policy_runner.py",
        home / "rsl_rl/rsl_rl/algorithms/ppo.py",
    ]
    reference_sources = [
        home / "SteadyTray/README.md",
        home / "SteadyTray/scripts/rsl_rl/adapter/adapter.py",
        home / "SteadyTray/scripts/rsl_rl/adapter/actor_critic.py",
        home / "SteadyTray/source/steadytray/steadytray/tasks/agents/rsl_rl_ppo_cfg.py",
        home / "SteadyTray/source/steadytray/steadytray/tasks/envs/steady_object_env_cfg.py",
    ]
    rows: list[dict[str, str]] = []
    for source in sources:
        if not source.is_file():
            continue
        relative = source.relative_to(home)
        destination = output / "source_snapshots" / relative
        copy_if_exists(source, destination)
        rows.append({"category": "project_source", "source": str(source), "copy": str(destination)})
    for source in reference_sources:
        if not source.is_file():
            continue
        relative = source.relative_to(home / "SteadyTray")
        destination = output / "references" / "SteadyTray" / relative
        copy_if_exists(source, destination)
        rows.append({"category": "reference_source", "source": str(source), "copy": str(destination)})
    return rows


def copy_images(output: Path) -> list[dict[str, str]]:
    home = Path("/home/zhaowenhao")
    rows: list[dict[str, str]] = []
    image_root = home / "图片"
    if image_root.is_dir():
        for source in sorted(path for path in image_root.iterdir() if path.is_file()):
            destination = output / "figures" / "source_images" / source.name
            copy_if_exists(source, destination)
            rows.append({"category": "source_image", "source": str(source), "copy": str(destination)})
    architecture_source = (
        home
        / ".codex/generated_images/019eca21-3d99-7e72-aa0f-071d59b70860"
        / "exec-528b91d2-4359-4a8d-8c13-7117926a9b95.png"
    )
    architecture_destination = output / "figures" / "architecture" / "s1_s2_residual_policy_architecture.png"
    if copy_if_exists(architecture_source, architecture_destination):
        rows.append(
            {
                "category": "generated_architecture",
                "source": str(architecture_source),
                "copy": str(architecture_destination),
            }
        )
    return rows


def build_source_image_contact_sheet(output: Path) -> None:
    source_root = output / "figures" / "source_images"
    files = sorted(path for path in source_root.iterdir() if path.is_file())
    if not files:
        return
    thumb_width, thumb_height, label_height, columns = 360, 240, 40, 4
    rows = math.ceil(len(files) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(files):
        try:
            with Image.open(path) as source:
                original_size = source.size
                image = source.convert("RGB")
            image.thumbnail((thumb_width - 16, thumb_height - 16))
            column, row = index % columns, index // columns
            x = column * thumb_width + (thumb_width - image.width) // 2
            y = row * (thumb_height + label_height) + (thumb_height - image.height) // 2
            sheet.paste(image, (x, y))
            label = f"{path.name}  {original_size[0]}x{original_size[1]}"
            draw.text((column * thumb_width + 8, row * (thumb_height + label_height) + thumb_height + 8), label, fill=(20, 20, 20))
        except Exception as exc:
            draw.text((8, index // columns * (thumb_height + label_height) + 8), f"{path.name}: {exc}", fill=(150, 0, 0))
    sheet.save(output / "figures" / "source_images_contact_sheet.png", quality=92)


def capture_git_state(project_root: Path, output: Path) -> None:
    repo_root = project_root.parent
    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    commands = {
        "isaaclab_nhb_git_status.txt": ["git", "-C", str(repo_root), "status", "--short"],
        "isaaclab_nhb_git_log.txt": [
            "git", "-C", str(repo_root), "log", "--date=iso", "--pretty=format:%h|%ad|%an|%s", "-100"
        ],
        "isaaclab_nhb_worktree.diff": ["git", "-C", str(repo_root), "diff"],
        "rsl_rl_git_status.txt": ["git", "-C", "/home/zhaowenhao/rsl_rl", "status", "--short"],
        "rsl_rl_worktree.diff": ["git", "-C", "/home/zhaowenhao/rsl_rl", "diff"],
    }
    for filename, command in commands.items():
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (metadata / filename).write_text(result.stdout, encoding="utf-8")


def write_timeline(run_rows: list[dict[str, Any]], output: Path) -> None:
    rows = [row for row in run_rows if row["experiment"] == "coopG1S1"]
    lines = [
        "# S1 调参时间线",
        "",
        "> 实验意图主要依据 run 文件夹命名推断；精确参数以 `configs/` 和 `metrics/run_summary.csv` 为准。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        error = row.get("final_Metrics__base_velocity__error_vel_xy")
        reward = row.get("final_Train__mean_reward")
        lines.extend(
            [
                f"## {index}. `{row['run']}`",
                "",
                f"- 推断目的：{row['run'].replace('_', ' ')}",
                f"- 环境数：`{row.get('num_envs')}`；配置迭代：`{row.get('max_iterations_cfg')}`；实际末步：`{row.get('actual_last_iteration')}`",
                f"- 初始化：resume=`{row.get('resume')}`，load_run=`{row.get('load_run')}`，checkpoint=`{row.get('load_checkpoint')}`",
                f"- 速度命令 X：`{row.get('command_lin_vel_x')}`；速度奖励：weight=`{row.get('track_vel_weight')}`，std=`{row.get('track_vel_std')}`",
                f"- 手部/步态：hand=`{row.get('hand_pose_weight')}`，arm=`{row.get('arm_pose_weight')}`，gait=`{row.get('gait_weight')}`",
                f"- Policy 含 base linear velocity：`{row.get('policy_has_base_lin_vel')}`",
                f"- 最终 XY 速度误差：`{error}`；最终 mean reward：`{reward}`",
                f"- 最新模型：`{row.get('latest_checkpoint_path')}`",
                "",
            ]
        )
    (output / "docs" / "02_s1_tuning_timeline.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "docs" / "02_s1_tuning_timeline.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output: Path, external_rows: list[dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    external_by_copy = {row["copy"]: row for row in external_rows}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        external = external_by_copy.get(str(path), {})
        rows.append(
            {
                "relative_path": str(path.relative_to(output)),
                "size_bytes": path.stat().st_size,
                "category": external.get("category", path.parts[len(output.parts)]),
                "source": external.get("source", "generated"),
            }
        )
    write_csv(output / "metadata" / "materials_manifest.csv", rows)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_root = project_root / "logs" / "rsl_rl"

    run_rows, checkpoint_rows, final_scalar_rows, series_by_run = collect_runs(log_root, output)
    plot_velocity_summary(run_rows, output)
    plot_selected_curves(series_by_run, output)
    plot_latest_dashboard(series_by_run, output)
    plot_reward_contributions(final_scalar_rows, output)
    external_rows = copy_sources(project_root, output) + copy_images(output)
    build_source_image_contact_sheet(output)
    external_rows += copy_hydra_outputs(project_root, output)
    capture_git_state(project_root, output)
    write_timeline(run_rows, output)
    write_manifest(output, external_rows)
    print(f"Report materials built at: {output}")
    print(f"Runs indexed: {len(run_rows)}")
    print(f"Checkpoints indexed: {len(checkpoint_rows)}")


if __name__ == "__main__":
    main()
