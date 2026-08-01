# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# 设置 CUDA 设备（必须在导入 CUDA 相关库之前设置）
import os

import argparse
import sys


from isaaclab.app import AppLauncher
import isaaclab_nhb 

# local imports
import cli_args  # isort: skip


USE_LOCAL_PLAY_DEFAULTS = True


def _project_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", path))


# Paste the exact checkpoint file here. Directories are intentionally rejected.
LOCAL_CHECKPOINT_PATH = _project_path(
    "logs/rsl_rl/coopG1S1/2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k/model_9999.pt"
)


LOCAL_PLAY_DEFAULTS = {
    "task": "CoopG1S1-29dof-HoldBox",
    "checkpoint": LOCAL_CHECKPOINT_PATH,
    "device": "cuda:5",
    "num_envs": 1,
    "show_s1_hand_targets": True,
    "real_time": False,
    "command": {
        "lin_vel_x": (0.3, 0.3),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
        "heading_command": False,
        "rel_standing_envs": 0.0,
    },
    "reset_base": {
        "yaw": (0.0, 0.0),
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
    },
}


_ORIGINAL_ARGV = sys.argv[1:]

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--csv_length", type=int, default=200, help="Length of CSV data recording (in steps).")
parser.add_argument(
    "--play_steps",
    type=int,
    default=None,
    help="Stop play cleanly after this many control steps (useful for plot/CSV export).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--show_s1_hand_targets",
    action="store_true",
    default=False,
    help="Visualize CoopG1S1 virtual hand target positions in the torso_link frame.",
)

parser.add_argument("--csv_export", action="store_true", default=False, help="Export data to CSV.")
parser.add_argument("--amp_stats", action="store_true", default=True, help="Enable AMP data statistics collection and output.")
parser.add_argument("--stats_interval", type=int, default=100, help="Interval (in steps) for AMP statistics output.")
parser.add_argument(
    "--plot_hand_reference_errors",
    action="store_true",
    default=False,
    help="Plot hand-reference tracking errors live during play.",
)
parser.add_argument(
    "--plot_hand_reference_trajectory",
    "--plot_hand_reference_trajectories",
    dest="plot_hand_reference_trajectory",
    action="store_true",
    default=False,
    help=(
        "Plot world-frame hand position/orientation and virtual-force "
        "target/actual/estimate curves live during play."
    ),
)
parser.add_argument(
    "--plot_policy_features",
    action="store_true",
    default=False,
    help=(
        "Record the trajectory-related inputs consumed by the S2 residual actor "
        "and export only the three dedicated policy-feature figures."
    ),
)
parser.add_argument(
    "--hand_reference_csv",
    type=str,
    default=None,
    help="Override the two-hand offline reference CSV configured by the selected task.",
)
parser.add_argument(
    "--show_hand_reference_in_sim",
    "--show_hand_reference_trajectory_in_sim",
    dest="show_hand_reference_in_sim",
    action="store_true",
    default=False,
    help="Draw the complete aligned hand-reference paths and actual hand trails in the Isaac Sim scene.",
)
parser.add_argument(
    "--hand_reference_vis_stride",
    type=int,
    default=5,
    help="Append one actual-path marker every N control steps in the Isaac Sim scene.",
)
parser.add_argument(
    "--hand_reference_vis_max_points",
    type=int,
    default=500,
    help="Maximum number of markers retained per reference/actual hand path.",
)
parser.add_argument(
    "--local_ground",
    action="store_true",
    default=False,
    help="Use a locally generated flat ground during play instead of the remote Isaac Sim grid USD.",
)
parser.add_argument(
    "--deterministic_eval",
    action="store_true",
    default=False,
    help="Disable observation corruption and use a fixed actuator delay during play evaluation.",
)
parser.add_argument(
    "--eval_actuator_delay_steps",
    type=int,
    default=1,
    help="Fixed actuator delay used with --deterministic_eval (in physics steps).",
)
parser.add_argument(
    "--diagnostic_csv",
    action="store_true",
    default=False,
    help="Record episode/reset, termination, root/torso state, per-axis hand errors, and action decomposition.",
)
parser.add_argument(
    "--hand_error_csv",
    action="store_true",
    default=False,
    help="Save per-step hand-reference tracking errors to CSV during play.",
)
parser.add_argument(
    "--hand_error_plot_window",
    type=int,
    default=500,
    help="Number of recent play steps shown in the live hand-error plot.",
)
parser.add_argument(
    "--hand_error_print_interval",
    type=int,
    default=0,
    help="Print hand-reference tracking errors every N play steps. Use 0 to disable.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()


def _cli_has_option(*option_names: str) -> bool:
    for arg in _ORIGINAL_ARGV:
        for option_name in option_names:
            if arg == option_name or arg.startswith(f"{option_name}="):
                return True
    return False


def _resolve_local_checkpoint_path(path: str) -> str:
    checkpoint_path = os.path.abspath(os.path.expanduser(path.strip().strip("'\"")))
    if not checkpoint_path.endswith(".pt"):
        raise ValueError(
            "LOCAL_CHECKPOINT_PATH must point to a .pt checkpoint file, not a directory or another file type: "
            f"{checkpoint_path}"
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"LOCAL_CHECKPOINT_PATH does not exist: {checkpoint_path}")
    return checkpoint_path


def _apply_local_play_defaults() -> None:
    if not USE_LOCAL_PLAY_DEFAULTS:
        return

    if not _cli_has_option("--task"):
        args_cli.task = LOCAL_PLAY_DEFAULTS["task"]
    if _cli_has_option("--checkpoint"):
        # Resolve local CLI paths before AppLauncher/Hydra can change any
        # process state.  Passing an unresolved local path through
        # retrieve_file_path() may make omni.client treat it as a remote URL
        # and return its download directory (for example, "/tmp").
        checkpoint_arg = str(args_cli.checkpoint).strip().strip("'\"")
        if "://" not in checkpoint_arg:
            args_cli.checkpoint = _resolve_local_checkpoint_path(checkpoint_arg)
    else:
        args_cli.checkpoint = _resolve_local_checkpoint_path(LOCAL_PLAY_DEFAULTS["checkpoint"])
    if not _cli_has_option("--device"):
        args_cli.device = LOCAL_PLAY_DEFAULTS["device"]
    if not _cli_has_option("--num_envs"):
        args_cli.num_envs = LOCAL_PLAY_DEFAULTS["num_envs"]
    if not _cli_has_option("--show_s1_hand_targets"):
        args_cli.show_s1_hand_targets = LOCAL_PLAY_DEFAULTS["show_s1_hand_targets"]
    if not _cli_has_option("--real-time"):
        args_cli.real_time = LOCAL_PLAY_DEFAULTS["real_time"]

    print(
        "[INFO] Local play defaults: "
        f"task={args_cli.task}, device={args_cli.device}, "
        f"num_envs={args_cli.num_envs}, checkpoint={args_cli.checkpoint}"
    )


_apply_local_play_defaults()




# 测试奖励用。不要在这里覆盖命令行参数；按 CLI 传入 task/checkpoint。
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/G1_ElevHist_ECMM_rough/2026-04-15_15-03-13_new_edge_pal_test2/model_24000.pt"
# args_cli.task = "G1-ElevHist-ECMM-Rough"

# ours
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/g1_mode13A5_rough/2026-02-05_11-39-36_mode13A5-NoAMP-colli500-edge5-Nostill/model_28000.pt"
# args_cli.task = "G1-Elevation-Net-Mode13A5-AMP-Rough"

# E1 光头
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/g1_E1/2026-02-05_14-00-07_E1-history-elevation/model_4000.pt"
# args_cli.task = "G1-terrain-E1"

# E2 不使用Rew的CNN
# args_cli.checkpoint = "/home/jyz/project/isaaclab_nhb_ws/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/g1_mode13A5_rough/2026-02-10_12-11-17_mode13A5-NobigAMP-E2-NoNewRew/model_28000.pt"
# args_cli.task = "G1-Elevation-Net-Mode13A5-AMP-Rough"

# BeyondMimic
# args_cli.checkpoint = "/home/andew/RL/RL_robot_ws/isaaclab_lyj/isaaclab_nhb/logs/rsl_rl/beyond_mimic_g1/2025-12-16_14-17-52_default/model_2000.pt"
# args_cli.task = "BeyondMimic-G1-Flat"

# 使用WebRTC串流
# args_cli.livestream=2

# Do not override the CLI device here; pass --device explicitly when launching.
# args_cli.headless = True
# args_cli.real_time = True
# args_cli.video = True
# args_cli.video_length = 710
# args_cli.csv_export = True
# args_cli.csv_length = 700


# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.headless:
    isaaclab_nhb.HEADLESS_FLAG = True 

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import csv
import json
import math
import os
import subprocess
import time
from collections import deque
import torch
import numpy as np

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# Import for getting camera position
import isaacsim.core.utils.stage as stage_utils
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_box_minus
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    try:
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    except ModuleNotFoundError:
        get_published_pretrained_checkpoint = None

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit
from rsl_rl.utils.vecenv_wrapper import RslRlVecEnvWrapperDictAction

from rsl_rl.utils.exporter import export_policy_as_onnx
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_nhb import * # 识别isaaclab_nhb库的内容
# from rsl_rl.utils.exporter import EstNetOnnxPolicyExporter, DWAQOnnxPolicyExporter  
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# PLACEHOLDER: Extension template (do not remove this comment)

from isaaclab_nhb.script.rsl_rl.data_logger import DataLogger
from isaaclab_nhb.script.rsl_rl.policy_feature_plotter import (
    APPENDED_BASE_ACTION_COLUMNS,
    BASE_COMMAND_COLUMNS,
    ERROR_HISTORY_COLUMNS,
    ESTIMATED_FORCE_COLUMNS,
    FORCE_CONTROL_AXIS_COLUMNS,
    GAIT_COLUMNS,
    REFERENCE_COLUMNS,
    TARGET_FORCE_COLUMNS,
    policy_feature_header,
)


HAND_KINEMATIC_ERROR_KEYS = (
    "left_pos_error_m",
    "right_pos_error_m",
    "mean_pos_error_m",
    "mean_rot_error_deg",
    "mean_lin_vel_error_mps",
    "mean_ang_vel_error_radps",
)
HAND_FORCE_ERROR_KEYS = (
    "mean_virtual_force_error_n",
    "mean_force_estimator_error_n",
)
HAND_REFERENCE_ERROR_KEYS = HAND_KINEMATIC_ERROR_KEYS + HAND_FORCE_ERROR_KEYS

HAND_REFERENCE_TRACKING_FIELDS = (
    ("position", ("x_m", "y_m", "z_m"), ("target", "actual")),
    ("quaternion", ("w", "x", "y", "z"), ("target", "actual")),
    ("linear_velocity", ("x_mps", "y_mps", "z_mps"), ("target", "actual")),
    ("angular_velocity", ("x_radps", "y_radps", "z_radps"), ("target", "actual")),
    # Force is environment-on-hand in world coordinates. The original CSV
    # hand-on-payload value is retained separately so the sign is auditable.
    ("force", ("x_n", "y_n", "z_n"), ("target", "actual", "estimated")),
    ("csv_hand_on_payload_force", ("x_n", "y_n", "z_n"), ("target",)),
    ("moment", ("x_nm", "y_nm", "z_nm"), ("target",)),
)


def _hand_reference_tracking_keys() -> tuple[str, ...]:
    keys = []
    for hand in ("left", "right"):
        for field, components, sources in HAND_REFERENCE_TRACKING_FIELDS:
            for source in sources:
                keys.extend(f"{hand}_{source}_{field}_{component}" for component in components)
    return tuple(keys)


HAND_REFERENCE_TRACKING_KEYS = _hand_reference_tracking_keys()
HAND_ERROR_PLOT_MIN_Y_SPAN = 0.1


def _read_hand_reference_error_csv(csv_path: str) -> dict[str, list[float]]:
    data = {"step": []}
    for key in HAND_REFERENCE_ERROR_KEYS:
        data[key] = []

    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row:
                continue
            data["step"].append(float(row["step"]))
            for key in HAND_REFERENCE_ERROR_KEYS:
                value = row.get(key)
                data[key].append(float(value) if value not in (None, "") else float("nan"))

    return data


def _hand_error_plot_upper_limit(
    *series: list[float],
    minimum_span: float = HAND_ERROR_PLOT_MIN_Y_SPAN,
) -> float:
    """Return a zero-based, 0.1-quantized axis range with a fixed minimum span."""
    finite_values = [
        value
        for values in series
        for value in values
        if math.isfinite(value)
    ]
    observed_max = max(finite_values, default=0.0)
    required = max(minimum_span, 1.05 * observed_max)
    return math.ceil(required / minimum_span) * minimum_span


def _save_hand_reference_error_plot(csv_path: str, output_path: str | None = None) -> str | None:
    if not csv_path or not os.path.isfile(csv_path):
        return None

    try:
        data = _read_hand_reference_error_csv(csv_path)
    except (OSError, ValueError, KeyError) as err:
        print(f"[WARNING] Failed to read hand-reference tracking CSV for plot export: {err}")
        return None

    steps = data["step"]
    if not steps:
        print("[WARNING] Hand-reference tracking CSV is empty; skipping plot export.")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as err:
        print(f"[WARNING] Failed to import matplotlib for hand-reference plot export: {err}")
        return None

    output_path = output_path or os.path.splitext(csv_path)[0] + ".png"
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    rotation_error_rad = [
        math.radians(value) for value in data["mean_rot_error_deg"]
    ]

    for axis in axes:
        axis.grid(True)

    axes[0].plot(steps, data["left_pos_error_m"], label="left")
    axes[0].plot(steps, data["right_pos_error_m"], label="right")
    axes[0].plot(steps, data["mean_pos_error_m"], label="mean", linewidth=2)
    axes[0].set_ylabel("pos error [m]")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(
        0.0,
        _hand_error_plot_upper_limit(
            data["left_pos_error_m"],
            data["right_pos_error_m"],
            data["mean_pos_error_m"],
        ),
    )

    axes[1].plot(steps, rotation_error_rad, color="tab:orange")
    axes[1].set_ylabel("rot error [rad]")
    axes[1].set_ylim(0.0, _hand_error_plot_upper_limit(rotation_error_rad))

    axes[2].plot(steps, data["mean_lin_vel_error_mps"], color="tab:blue")
    axes[2].set_ylabel("lin vel error [m/s]")
    axes[2].set_ylim(
        0.0,
        _hand_error_plot_upper_limit(data["mean_lin_vel_error_mps"]),
    )

    axes[3].plot(steps, data["mean_ang_vel_error_radps"], color="tab:orange")
    axes[3].set_ylabel("ang vel error [rad/s]")
    axes[3].set_ylim(
        0.0,
        _hand_error_plot_upper_limit(data["mean_ang_vel_error_radps"]),
    )

    axes[4].plot(steps, data["mean_virtual_force_error_n"], label="tracking")
    axes[4].plot(steps, data["mean_force_estimator_error_n"], label="estimator")
    axes[4].set_ylabel("force error [N]")
    axes[4].set_xlabel("play step")
    axes[4].legend(loc="upper right")
    axes[4].set_ylim(
        0.0,
        _hand_error_plot_upper_limit(
            data["mean_virtual_force_error_n"],
            data["mean_force_estimator_error_n"],
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


class HandReferenceErrorMonitor:
    """Record and visualize S2 hand-reference tracking during play."""

    def __init__(
        self,
        env,
        evidence_dir: str,
        enable_error_plot: bool = False,
        enable_trajectory_plot: bool = False,
        enable_csv: bool = False,
        enable_diagnostics: bool = False,
        window: int = 500,
        print_interval: int = 50,
        metadata: dict | None = None,
        timestamp: str | None = None,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.enable_error_plot = enable_error_plot
        self.enable_trajectory_plot = enable_trajectory_plot
        self.enable_diagnostics = enable_diagnostics
        self.enable_csv = enable_csv or enable_diagnostics
        self.window = max(2, int(window))
        self.print_interval = int(print_interval)
        self.step_dt = float(self.unwrapped.step_dt)
        self.step_count = 0
        self.episode_id = 0
        self.episode_step = 0
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self.metadata_path = None
        self.episode_summary_path = None
        self.png_path = None
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        self.rawdata_dir = os.path.join(evidence_dir, "rawdata")
        self.pics_dir = os.path.join(evidence_dir, "pics")
        self.plot_processes = []
        self.episode_summaries = []
        self._reset_episode_accumulator()

        try:
            self.command_term = self.unwrapped.command_manager.get_term("hand_reference")
        except Exception as err:
            raise RuntimeError("Task does not expose a hand_reference command term.") from err

        self.has_tracking_state = hasattr(self.command_term, "tracking_state")
        if self.enable_trajectory_plot and not self.has_tracking_state:
            raise RuntimeError("hand_reference command does not expose reference/actual tracking states.")
        self.tracking_state_getter = getattr(
            self.command_term,
            "tracking_state_world",
            getattr(self.command_term, "tracking_state", None),
        )
        self.virtual_spring = getattr(self.unwrapped, "_virtual_spring", None)

        self.robot = self.unwrapped.scene["robot"]
        try:
            self.torso_id = self.robot.body_names.index("torso_link")
        except ValueError as err:
            raise RuntimeError("Robot has no torso_link required by play diagnostics.") from err

        action_dim = int(self.unwrapped.action_manager.total_action_dim)
        joint_names = []
        action_terms = getattr(self.unwrapped.action_manager, "_terms", {})
        for action_term in action_terms.values():
            term_joint_names = getattr(action_term, "_joint_names", None)
            if term_joint_names is not None:
                joint_names.extend(term_joint_names)
        if len(joint_names) != action_dim:
            joint_names = [f"index_{index:02d}" for index in range(action_dim)]
        self.action_names = [
            f"{index:02d}_{''.join(character if character.isalnum() else '_' for character in name)}"
            for index, name in enumerate(joint_names)
        ]
        self.diagnostic_keys = self._diagnostic_keys()

        self.enable_csv = self.enable_csv or self.enable_error_plot or self.enable_trajectory_plot
        if self.enable_csv:
            os.makedirs(self.rawdata_dir, exist_ok=True)
            os.makedirs(self.pics_dir, exist_ok=True)
            self.csv_path = os.path.join(self.rawdata_dir, f"tracking_{self.timestamp}.csv")
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            tracking_keys = HAND_REFERENCE_TRACKING_KEYS if self.has_tracking_state else ()
            diagnostic_keys = self.diagnostic_keys if self.enable_diagnostics else ()
            self.csv_writer.writerow(
                ("step", "time_s", *diagnostic_keys, *HAND_REFERENCE_ERROR_KEYS, *tracking_keys)
            )
            print(f"[INFO] Saving hand-reference tracking errors to: {self.csv_path}")
            if self.enable_diagnostics:
                self.metadata_path = os.path.splitext(self.csv_path)[0] + "_metadata.json"
                diagnostic_metadata = dict(metadata or {})
                diagnostic_metadata.update(
                    {
                        "coordinate_frame": "world",
                        "step_dt_s": self.step_dt,
                        "action_names": self.action_names,
                        "termination_terms": list(self.unwrapped.termination_manager.active_terms),
                        "done_rows_contain_post_reset_state": True,
                    }
                )
                with open(self.metadata_path, "w") as metadata_file:
                    json.dump(diagnostic_metadata, metadata_file, indent=2, ensure_ascii=False)
                print(f"[INFO] Saving deterministic-play metadata to: {self.metadata_path}")

        if self.enable_error_plot:
            plotter_path = os.path.join(os.path.dirname(__file__), "hand_error_plotter.py")
            self.plot_processes.append(subprocess.Popen(
                [
                    sys.executable,
                    plotter_path,
                    "--csv",
                    self.csv_path,
                    "--window",
                    str(self.window),
                ]
            ))
            print("[INFO] Live hand-reference error plot enabled in a separate process.")

        if self.enable_trajectory_plot:
            plotter_path = os.path.join(os.path.dirname(__file__), "hand_tracking_plotter.py")
            plotter_command = [
                sys.executable,
                plotter_path,
                "--csv",
                self.csv_path,
                "--window",
                str(self.window),
            ]
            if self.enable_diagnostics:
                plotter_command.append("--include-diagnostics")
            self.plot_processes.append(subprocess.Popen(plotter_command))
            print("[INFO] Live world-frame reference-vs-actual hand trajectory plots enabled in a separate process.")

    def _reset_episode_accumulator(self) -> None:
        self.current_episode_values = {key: [] for key in HAND_REFERENCE_ERROR_KEYS}
        self.current_episode_diagnostic_values = {
            key: []
            for key in (
                "root_position_world_x_m",
                "root_position_world_y_m",
                "root_position_world_z_m",
                "root_rpy_world_yaw_deg",
                "torso_height_world_m",
                "base_action_l2",
                "residual_action_l2",
                "residual_action_max_abs",
                "total_action_l2",
            )
        }

    def _diagnostic_keys(self) -> tuple[str, ...]:
        keys = [
            "episode_id",
            "episode_step",
            "episode_time_s",
            "done",
            "terminated",
            "time_out",
            "termination_reason",
            "state_is_post_reset",
        ]
        for body in ("root", "torso"):
            keys.extend(f"{body}_position_world_{axis}_m" for axis in "xyz")
            keys.extend(f"{body}_rpy_world_{axis}_deg" for axis in ("roll", "pitch", "yaw"))
            keys.extend(f"{body}_linear_velocity_world_{axis}_mps" for axis in "xyz")
            keys.extend(f"{body}_angular_velocity_world_{axis}_radps" for axis in "xyz")
        keys.append("torso_height_world_m")
        for hand in ("left", "right"):
            keys.extend(f"{hand}_position_error_world_{axis}_m" for axis in "xyz")
            keys.extend(f"{hand}_rotation_error_world_{axis}_deg" for axis in "xyz")
            keys.extend(f"{hand}_linear_velocity_error_world_{axis}_mps" for axis in "xyz")
            keys.extend(f"{hand}_angular_velocity_error_world_{axis}_radps" for axis in "xyz")
        keys.extend(
            (
                "gait_frequency_hz",
                "gait_phase",
                "gait_sin",
                "gait_cos",
                "gait_sin_right",
                "gait_cos_right",
                "base_command_x_mps",
                "base_command_y_mps",
                "base_command_yaw_radps",
            )
        )
        keys.extend(("base_action_l2", "residual_action_l2", "residual_action_max_abs", "total_action_l2"))
        for source in ("base", "residual", "total"):
            keys.extend(f"{source}_action_{name}" for name in self.action_names)
        return tuple(keys)

    @staticmethod
    def _tensor_values(tensor: torch.Tensor) -> tuple[float, ...]:
        return tuple(float(value) for value in tensor.detach().reshape(-1).cpu().tolist())

    def _body_diagnostic_values(self) -> dict[str, float]:
        root_pos = self.robot.data.root_pos_w[0]
        root_quat = self.robot.data.root_quat_w[0]
        root_lin_vel = self.robot.data.root_lin_vel_w[0]
        root_ang_vel = self.robot.data.root_ang_vel_w[0]
        torso_pos = self.robot.data.body_link_pos_w[0, self.torso_id]
        torso_quat = self.robot.data.body_link_quat_w[0, self.torso_id]
        torso_lin_vel = self.robot.data.body_link_lin_vel_w[0, self.torso_id]
        torso_ang_vel = self.robot.data.body_link_ang_vel_w[0, self.torso_id]

        values = {}
        for body, pos, quat, lin_vel, ang_vel in (
            ("root", root_pos, root_quat, root_lin_vel, root_ang_vel),
            ("torso", torso_pos, torso_quat, torso_lin_vel, torso_ang_vel),
        ):
            roll, pitch, yaw = euler_xyz_from_quat(quat.unsqueeze(0))
            for axis, value in zip("xyz", self._tensor_values(pos)):
                values[f"{body}_position_world_{axis}_m"] = value
            for axis, value in zip(
                ("roll", "pitch", "yaw"),
                self._tensor_values(torch.stack((roll[0], pitch[0], yaw[0])) * (180.0 / math.pi)),
            ):
                values[f"{body}_rpy_world_{axis}_deg"] = value
            for axis, value in zip("xyz", self._tensor_values(lin_vel)):
                values[f"{body}_linear_velocity_world_{axis}_mps"] = value
            for axis, value in zip("xyz", self._tensor_values(ang_vel)):
                values[f"{body}_angular_velocity_world_{axis}_radps"] = value
        values["torso_height_world_m"] = float(torso_pos[2].detach().cpu())
        return values

    def _hand_error_component_values(self, state: dict[str, torch.Tensor]) -> dict[str, float]:
        values = {}
        position_error = state["target_position"] - state["actual_position"]
        rotation_error = quat_box_minus(
            state["target_quaternion"].reshape(-1, 4),
            state["actual_quaternion"].reshape(-1, 4),
        ).reshape(1, 2, 3) * (180.0 / math.pi)
        linear_velocity_error = state["target_linear_velocity"] - state["actual_linear_velocity"]
        angular_velocity_error = state["target_angular_velocity"] - state["actual_angular_velocity"]
        for hand_index, hand in enumerate(("left", "right")):
            for field, tensor, unit in (
                ("position", position_error, "m"),
                ("rotation", rotation_error, "deg"),
                ("linear_velocity", linear_velocity_error, "mps"),
                ("angular_velocity", angular_velocity_error, "radps"),
            ):
                for axis, value in zip("xyz", self._tensor_values(tensor[0, hand_index])):
                    values[f"{hand}_{field}_error_world_{axis}_{unit}"] = value
        return values

    def _command_diagnostic_values(self) -> dict[str, float]:
        values = {
            key: float("nan")
            for key in (
                "gait_frequency_hz",
                "gait_phase",
                "gait_sin",
                "gait_cos",
                "gait_sin_right",
                "gait_cos_right",
                "base_command_x_mps",
                "base_command_y_mps",
                "base_command_yaw_radps",
            )
        }
        try:
            gait = self.unwrapped.command_manager.get_command("gait_command")[0]
            gait_values = self._tensor_values(gait)
            if len(gait_values) >= 7:
                values.update(
                    {
                        "gait_frequency_hz": gait_values[2],
                        "gait_phase": (math.atan2(gait_values[3], gait_values[4]) / (2.0 * math.pi)) % 1.0,
                        "gait_sin": gait_values[3],
                        "gait_cos": gait_values[4],
                        "gait_sin_right": gait_values[5],
                        "gait_cos_right": gait_values[6],
                    }
                )
        except (AttributeError, KeyError, RuntimeError, ValueError):
            pass

        try:
            base_command = self.unwrapped.command_manager.get_command("base_velocity")[0]
            base_values = self._tensor_values(base_command)
            for key, value in zip(
                ("base_command_x_mps", "base_command_y_mps", "base_command_yaw_radps"),
                base_values,
            ):
                values[key] = value
        except (AttributeError, KeyError, RuntimeError, ValueError):
            pass
        return values

    def _action_values(
        self,
        total_action: torch.Tensor | None,
        base_action: torch.Tensor | None,
        residual_action: torch.Tensor | None,
    ) -> dict[str, float]:
        values = {}
        for source, tensor in (
            ("base", base_action),
            ("residual", residual_action),
            ("total", total_action),
        ):
            source_values = [float("nan")] * len(self.action_names)
            if tensor is not None:
                flat_values = self._tensor_values(tensor[0])
                source_values[: min(len(source_values), len(flat_values))] = flat_values[: len(source_values)]
            for name, value in zip(self.action_names, source_values):
                values[f"{source}_action_{name}"] = value
        for source, tensor in (("base", base_action), ("residual", residual_action), ("total", total_action)):
            if tensor is None:
                values[f"{source}_action_l2"] = float("nan")
            else:
                values[f"{source}_action_l2"] = float(torch.linalg.vector_norm(tensor[0]).detach().cpu())
        values["residual_action_max_abs"] = (
            float(torch.max(torch.abs(residual_action[0])).detach().cpu())
            if residual_action is not None
            else float("nan")
        )
        return values

    def _finish_episode(self, completed: bool, termination_reason: str) -> None:
        summary = {
            "episode_id": self.episode_id,
            "completed": int(completed),
            "steps": self.episode_step,
            "duration_s": self.episode_step * self.step_dt,
            "termination_reason": termination_reason,
        }
        for key in HAND_REFERENCE_ERROR_KEYS:
            episode_values = self.current_episode_values[key]
            summary[f"mean_{key}"] = (
                sum(episode_values) / len(episode_values) if episode_values else float("nan")
            )
            summary[f"max_{key}"] = max(episode_values) if episode_values else float("nan")
        diagnostics = self.current_episode_diagnostic_values
        for axis in "xyz":
            key = f"root_position_world_{axis}_m"
            samples = diagnostics[key]
            summary[f"root_start_{axis}_m"] = samples[0] if samples else float("nan")
            summary[f"root_end_{axis}_m"] = samples[-1] if samples else float("nan")
            summary[f"root_delta_{axis}_m"] = samples[-1] - samples[0] if samples else float("nan")
        delta_x = summary["root_delta_x_m"]
        delta_y = summary["root_delta_y_m"]
        summary["root_xy_displacement_heading_deg"] = math.degrees(math.atan2(delta_y, delta_x))
        yaw_samples = diagnostics["root_rpy_world_yaw_deg"]
        torso_height_samples = diagnostics["torso_height_world_m"]
        summary["max_abs_root_yaw_deg"] = max((abs(value) for value in yaw_samples), default=float("nan"))
        summary["min_torso_height_m"] = min(torso_height_samples, default=float("nan"))
        for key in ("base_action_l2", "residual_action_l2", "residual_action_max_abs", "total_action_l2"):
            samples = diagnostics[key]
            summary[f"mean_{key}"] = sum(samples) / len(samples) if samples else float("nan")
            summary[f"max_{key}"] = max(samples) if samples else float("nan")
        self.episode_summaries.append(summary)
        status = termination_reason if completed else "incomplete_play_stop"
        print(
            "[EpisodeSummary] "
            f"episode={self.episode_id} steps={self.episode_step} duration={summary['duration_s']:.2f}s "
            f"status={status} mean_pos={summary['mean_mean_pos_error_m']:.4f}m "
            f"max_pos={summary['max_mean_pos_error_m']:.4f}m "
            f"mean_rot={summary['mean_mean_rot_error_deg']:.2f}deg "
            f"max_rot={summary['max_mean_rot_error_deg']:.2f}deg "
            f"mean_force={summary['mean_mean_virtual_force_error_n']:.4f}N "
            f"mean_force_est={summary['mean_mean_force_estimator_error_n']:.4f}N "
            f"travel_heading={summary['root_xy_displacement_heading_deg']:.2f}deg "
            f"max_abs_yaw={summary['max_abs_root_yaw_deg']:.2f}deg "
            f"min_torso_height={summary['min_torso_height_m']:.3f}m"
        )

    def update(
        self,
        sim_time_s: float,
        done: bool = False,
        terminated: bool = False,
        time_out: bool = False,
        termination_reason: str = "",
        total_action: torch.Tensor | None = None,
        base_action: torch.Tensor | None = None,
        residual_action: torch.Tensor | None = None,
        estimated_force: torch.Tensor | None = None,
    ) -> None:
        self.episode_step += 1
        latest = self.command_term.latest_tracking_errors
        values = {
            key: float(latest[key][0].detach().cpu())
            for key in HAND_KINEMATIC_ERROR_KEYS
        }
        values.update({key: float("nan") for key in HAND_FORCE_ERROR_KEYS})
        if self.virtual_spring is not None:
            force_error = torch.norm(
                self.virtual_spring.target_force_robot_w[0]
                - self.virtual_spring.actual_virtual_force_robot_w[0],
                dim=-1,
            ).mean()
            values["mean_virtual_force_error_n"] = float(force_error.detach().cpu())
            if estimated_force is not None:
                actual_force_t = self.virtual_spring.actual_force_observation()[0]
                estimator_error = torch.norm(
                    estimated_force[0].reshape(2, 3)
                    - actual_force_t.reshape(2, 3),
                    dim=-1,
                ).mean()
                values["mean_force_estimator_error_n"] = float(
                    estimator_error.detach().cpu()
                )
        # Isaac Lab automatically resets inside env.step().  A done row therefore
        # contains the reset state, not the terminal state.  Write NaNs to create
        # an explicit plot break; the preceding row is the last physical sample.
        if done:
            values = {key: float("nan") for key in HAND_REFERENCE_ERROR_KEYS}
        else:
            for key, value in values.items():
                self.current_episode_values[key].append(value)

        tracking_values = {}
        if self.has_tracking_state:
            state = self.tracking_state_getter([0])
            csv_force_w = state["target_force"]
            state["target_csv_hand_on_payload_force"] = csv_force_w
            if self.virtual_spring is not None:
                state["target_force"] = self.virtual_spring.target_force_robot_w[[0]]
                state["actual_force"] = self.virtual_spring.actual_virtual_force_robot_w[[0]]
            else:
                state["actual_force"] = torch.full_like(csv_force_w, float("nan"))

            if estimated_force is not None:
                estimated_force_t = estimated_force[[0]].reshape(1, 2, 3)
                torso_quat_w = self.robot.data.body_link_quat_w[[0], self.torso_id]
                torso_quat_hands = torso_quat_w.unsqueeze(1).expand(-1, 2, -1)
                state["estimated_force"] = quat_apply(
                    torso_quat_hands.reshape(-1, 4),
                    estimated_force_t.reshape(-1, 3),
                ).reshape(1, 2, 3)
            else:
                state["estimated_force"] = torch.full_like(csv_force_w, float("nan"))

            for hand_index, hand in enumerate(("left", "right")):
                for field, components, sources in HAND_REFERENCE_TRACKING_FIELDS:
                    for source in sources:
                        tensor = state[f"{source}_{field}"][0, hand_index].detach().cpu()
                        for component, value in zip(components, tensor.tolist()):
                            tracking_values[f"{hand}_{source}_{field}_{component}"] = float(value)
            if done:
                tracking_values = {key: float("nan") for key in HAND_REFERENCE_TRACKING_KEYS}

        diagnostic_values = {}
        if self.enable_diagnostics:
            diagnostic_values = {
                "episode_id": self.episode_id,
                "episode_step": self.episode_step,
                "episode_time_s": self.episode_step * self.step_dt,
                "done": int(done),
                "terminated": int(terminated),
                "time_out": int(time_out),
                "termination_reason": termination_reason,
                "state_is_post_reset": int(done),
            }
            diagnostic_values.update(self._body_diagnostic_values())
            diagnostic_values.update(self._command_diagnostic_values())
            if self.has_tracking_state and not done:
                diagnostic_values.update(self._hand_error_component_values(state))
            else:
                for key in self.diagnostic_keys:
                    if "_error_world_" in key:
                        diagnostic_values[key] = float("nan")
            diagnostic_values.update(self._action_values(total_action, base_action, residual_action))
            if not done:
                for key in self.current_episode_diagnostic_values:
                    self.current_episode_diagnostic_values[key].append(float(diagnostic_values[key]))

        if self.csv_writer is not None:
            tracking_row = (
                tuple(tracking_values[key] for key in HAND_REFERENCE_TRACKING_KEYS)
                if self.has_tracking_state
                else ()
            )
            diagnostic_row = (
                tuple(diagnostic_values[key] for key in self.diagnostic_keys)
                if self.enable_diagnostics
                else ()
            )
            self.csv_writer.writerow(
                (
                    self.step_count,
                    sim_time_s,
                    *diagnostic_row,
                    *(values[key] for key in HAND_REFERENCE_ERROR_KEYS),
                    *tracking_row,
                )
            )
            self.csv_file.flush()

        if not done and self.print_interval > 0 and self.step_count % self.print_interval == 0:
            force_text = ""
            if self.virtual_spring is not None:
                force_text = f" force={values['mean_virtual_force_error_n']:.4f} N"
                if estimated_force is not None:
                    force_text += (
                        f" force_est={values['mean_force_estimator_error_n']:.4f} N"
                    )
            print(
                "[HandRef] "
                f"step={self.step_count} "
                f"pos={values['mean_pos_error_m']:.4f} m "
                f"rot={values['mean_rot_error_deg']:.2f} deg "
                f"lin_vel={values['mean_lin_vel_error_mps']:.4f} m/s "
                f"ang_vel={values['mean_ang_vel_error_radps']:.4f} rad/s"
                f"{force_text}"
            )

        self.step_count += 1
        if done:
            self._finish_episode(completed=True, termination_reason=termination_reason or "unknown")
            self.episode_id += 1
            self.episode_step = 0
            self._reset_episode_accumulator()

    def _write_episode_summaries(self) -> None:
        if not self.enable_diagnostics or self.csv_path is None or not self.episode_summaries:
            return
        self.episode_summary_path = os.path.splitext(self.csv_path)[0] + "_episodes.csv"
        with open(self.episode_summary_path, "w", newline="") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=list(self.episode_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(self.episode_summaries)
        print(f"[INFO] Episode diagnostics saved to: {self.episode_summary_path}")

    def close(self) -> None:
        if self.enable_diagnostics and self.episode_step > 0:
            self._finish_episode(completed=False, termination_reason="incomplete_play_stop")
        self._write_episode_summaries()
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"[INFO] Hand-reference tracking errors saved to: {self.csv_path}")
            if self.enable_error_plot:
                error_plot_path = os.path.join(self.pics_dir, f"tracking_error_{self.timestamp}.png")
                self.png_path = _save_hand_reference_error_plot(self.csv_path, error_plot_path)
                if self.png_path is not None:
                    print(f"[INFO] Hand-reference tracking plot saved to: {self.png_path}")
        for plot_process in self.plot_processes:
            if plot_process.poll() is None:
                plot_process.terminate()
        if self.enable_trajectory_plot and self.csv_path is not None:
            plotter_path = os.path.join(os.path.dirname(__file__), "hand_tracking_plotter.py")
            result = subprocess.run(
                [
                    sys.executable,
                    plotter_path,
                    "--csv",
                    self.csv_path,
                    "--save-only",
                    "--output-dir",
                    self.pics_dir,
                    "--rawdata-dir",
                    self.rawdata_dir,
                    "--timestamp",
                    self.timestamp,
                ],
                check=False,
            )
            if result.returncode != 0:
                print("[WARNING] Failed to export reference-vs-actual hand trajectory plots.")


class PolicyFeatureMonitor:
    """Record the exact action-time trajectory features consumed by the S2 actor."""

    def __init__(
        self,
        env,
        policy_module,
        evidence_dir: str,
        metadata: dict | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.env = env.unwrapped
        self.policy_module = policy_module
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        self.rawdata_dir = os.path.join(evidence_dir, "rawdata")
        self.pics_dir = os.path.join(evidence_dir, "pics")
        os.makedirs(self.rawdata_dir, exist_ok=True)
        os.makedirs(self.pics_dir, exist_ok=True)

        self.residual_slices = self._observation_term_slices("residual_policy")
        self.force_context_slices = self._observation_term_slices("force_context")
        self._validate_term_dim(self.residual_slices, "hand_reference", 26)
        self._validate_term_dim(self.residual_slices, "hand_reference_error", 120)
        self._validate_term_dim(self.residual_slices, "velocity_commands", 3)
        self._validate_term_dim(self.residual_slices, "gait_commands", 7)
        self._validate_term_dim(self.force_context_slices, "target_virtual_force", 6)
        self._validate_term_dim(self.force_context_slices, "force_control_axes", 6)

        required_policy_methods = ("_compute_base_action", "_actor_input")
        missing_methods = [
            method for method in required_policy_methods if not callable(getattr(policy_module, method, None))
        ]
        if missing_methods:
            raise RuntimeError(
                "--plot_policy_features requires ActorCriticResidual methods "
                f"{required_policy_methods}; missing {missing_methods}."
            )
        active_action_indices = getattr(policy_module, "active_action_indices", None)
        if active_action_indices is None or int(active_action_indices.numel()) != len(APPENDED_BASE_ACTION_COLUMNS):
            raise RuntimeError(
                "S2 appended base-action layout mismatch: expected "
                f"{len(APPENDED_BASE_ACTION_COLUMNS)} active arm actions."
            )
        self.active_action_indices = active_action_indices
        self.actor_input_dim = int(getattr(policy_module, "num_residual_actor_obs", 0))
        if self.actor_input_dim <= 0:
            raise RuntimeError("S2 policy does not expose a positive num_residual_actor_obs.")

        self.csv_path = os.path.join(self.rawdata_dir, f"policy_features_{self.timestamp}.csv")
        self.metadata_path = os.path.join(
            self.rawdata_dir, f"policy_features_{self.timestamp}_metadata.json"
        )
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(policy_feature_header(self.actor_input_dim))
        self.step_count = 0
        self.episode_id = 0

        diagnostic_metadata = dict(metadata or {})
        diagnostic_metadata.update(
            {
                "sample_semantics": "pre-action observation obs_t used to compute action_t",
                "coordinate_frame": "current torso_link frame",
                "step_dt_s": float(self.env.step_dt),
                "trajectory_preview_steps": int(
                    getattr(self.env.cfg.commands.hand_reference, "preview_steps", 0)
                ),
                "history_order_in_csv": ["lag4", "lag3", "lag2", "lag1", "lag0"],
                "history_semantics": "oldest to newest; lag0 is the current observation",
                "orientation_reference_storage": "scalar-first quaternion wxyz",
                "orientation_error_storage": "target-minus-actual axis-angle vector in radians",
                "actual_virtual_force_is_policy_input": False,
                "actor_input_dim": self.actor_input_dim,
                "actor_input_layout": {
                    "normalized_control_features": [
                        0,
                        int(getattr(policy_module, "num_residual_control_obs", 0)),
                    ],
                    "estimated_virtual_force_scaled": [
                        int(getattr(policy_module, "num_residual_control_obs", 0)),
                        self.actor_input_dim,
                    ],
                },
                "force_estimator_scale": float(getattr(policy_module, "force_estimator_scale", 1.0)),
                "policy_schema_version": int(
                    getattr(policy_module, "policy_schema_version", torch.tensor(-1)).item()
                ),
                "observation_schema": str(getattr(policy_module, "observation_schema", "unknown")),
                "observation_terms": {
                    group_name: [
                        {"name": name, "shape": [int(dimension) for dimension in shape]}
                        for name, shape in zip(
                            self.env.observation_manager.active_terms[group_name],
                            self.env.observation_manager.group_obs_term_dim[group_name],
                        )
                    ]
                    for group_name in ("residual_policy", "force_context")
                },
                "semantic_columns": {
                    "reference": list(REFERENCE_COLUMNS),
                    "error_history": list(ERROR_HISTORY_COLUMNS),
                    "target_virtual_force": list(TARGET_FORCE_COLUMNS),
                    "force_control_axes": list(FORCE_CONTROL_AXIS_COLUMNS),
                    "gait": list(GAIT_COLUMNS),
                    "base_command": list(BASE_COMMAND_COLUMNS),
                    "estimated_virtual_force": list(ESTIMATED_FORCE_COLUMNS),
                    "appended_base_action": list(APPENDED_BASE_ACTION_COLUMNS),
                },
            }
        )
        with open(self.metadata_path, "w") as metadata_file:
            json.dump(diagnostic_metadata, metadata_file, indent=2, ensure_ascii=False)
        print(f"[INFO] Saving exact S2 policy trajectory features to: {self.csv_path}")
        print(f"[INFO] Saving S2 policy-feature metadata to: {self.metadata_path}")

    def _observation_term_slices(self, group_name: str) -> dict[str, slice]:
        manager = self.env.observation_manager
        if group_name not in manager.active_terms:
            raise RuntimeError(
                f"S2 observation group '{group_name}' is missing; available groups: "
                f"{list(manager.active_terms)}"
            )
        names = manager.active_terms[group_name]
        dimensions = manager.group_obs_term_dim[group_name]
        offset = 0
        slices = {}
        for name, dimension in zip(names, dimensions):
            size = math.prod(dimension)
            slices[name] = slice(offset, offset + size)
            offset += size
        return slices

    @staticmethod
    def _validate_term_dim(slices: dict[str, slice], term_name: str, expected_dim: int) -> None:
        term_slice = slices.get(term_name)
        if term_slice is None:
            raise RuntimeError(f"Required S2 policy observation term is missing: {term_name}")
        actual_dim = term_slice.stop - term_slice.start
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"S2 policy term '{term_name}' has {actual_dim} dimensions; expected {expected_dim}."
            )

    @staticmethod
    def _values(tensor: torch.Tensor) -> list[float]:
        return tensor.detach().reshape(-1).cpu().tolist()

    def record(self, obs, time_s: float) -> None:
        if "residual_policy" not in obs or "force_context" not in obs:
            raise RuntimeError(
                "Policy observation is missing residual_policy or force_context; "
                f"available groups: {list(obs.keys())}"
            )
        residual_obs = obs["residual_policy"][0]
        force_context = obs["force_context"][0]

        reference = residual_obs[self.residual_slices["hand_reference"]]
        error_history = residual_obs[self.residual_slices["hand_reference_error"]]
        gait = residual_obs[self.residual_slices["gait_commands"]]
        base_command = residual_obs[self.residual_slices["velocity_commands"]]
        target_force = force_context[self.force_context_slices["target_virtual_force"]]
        force_control_axes = force_context[self.force_context_slices["force_control_axes"]]

        base_action = self.policy_module._compute_base_action(obs)
        actor_input, estimated_force_scaled = self.policy_module._actor_input(obs, base_action)
        appended_base_action = torch.index_select(
            base_action,
            dim=-1,
            index=self.active_action_indices,
        )[0]
        estimated_force = estimated_force_scaled[0] * float(
            getattr(self.policy_module, "force_estimator_scale", 1.0)
        )
        if actor_input.shape[-1] != self.actor_input_dim:
            raise RuntimeError(
                f"S2 actor input has {actor_input.shape[-1]} dimensions; expected {self.actor_input_dim}."
            )

        episode_step = int(self.env.episode_length_buf[0].detach().cpu())
        row = (
            self.step_count,
            float(time_s),
            self.episode_id,
            episode_step,
            *self._values(reference),
            *self._values(error_history),
            *self._values(target_force),
            *self._values(force_control_axes),
            *self._values(gait),
            *self._values(base_command),
            *self._values(estimated_force),
            *self._values(appended_base_action),
            *self._values(actor_input[0]),
        )
        expected_columns = len(policy_feature_header(self.actor_input_dim))
        if len(row) != expected_columns:
            raise RuntimeError(
                f"Policy-feature row contains {len(row)} values; expected {expected_columns}."
            )
        self.csv_writer.writerow(row)
        self.step_count += 1

    def after_step(self, done: bool) -> None:
        if done:
            self.episode_id += 1

    def close(self) -> None:
        if self.csv_file is None:
            return
        self.csv_file.close()
        self.csv_file = None
        print(f"[INFO] S2 policy trajectory features saved to: {self.csv_path}")
        plotter_path = os.path.join(os.path.dirname(__file__), "policy_feature_plotter.py")
        result = subprocess.run(
            [
                sys.executable,
                plotter_path,
                "--csv",
                self.csv_path,
                "--output-dir",
                self.pics_dir,
                "--rawdata-dir",
                self.rawdata_dir,
                "--timestamp",
                self.timestamp,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Failed to export the three S2 policy-feature plots.")


def _make_hand_path_marker(
    prim_path: str,
    color: tuple[float, float, float],
    radius: float,
) -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "marker": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
            },
        )
    )


class HandReferenceInSimVisualizer:
    """Draw complete reference paths and growing actual trails inside Isaac Sim."""

    def __init__(self, env, stride: int = 5, max_points: int = 500):
        if stride <= 0:
            raise ValueError("--hand_reference_vis_stride must be positive.")
        if max_points <= 1:
            raise ValueError("--hand_reference_vis_max_points must be greater than one.")

        self.env = env.unwrapped
        self.stride = int(stride)
        self.max_points = int(max_points)
        self.command_term = self.env.command_manager.get_term("hand_reference")
        if not hasattr(self.command_term, "tracking_state"):
            raise RuntimeError("hand_reference command does not expose tracking_state().")
        if not hasattr(self.command_term, "reference_path_world"):
            raise RuntimeError("hand_reference command does not expose reference_path_world().")

        self.robot = self.env.scene["robot"]
        self.torso_id = self.robot.body_names.index("torso_link")
        self.actual_paths = (deque(maxlen=self.max_points), deque(maxlen=self.max_points))

        self.reference_markers = (
            _make_hand_path_marker("/Visuals/HandReference/LeftReferencePath", (0.15, 1.0, 0.15), 0.012),
            _make_hand_path_marker("/Visuals/HandReference/RightReferencePath", (1.0, 0.15, 1.0), 0.012),
        )
        self.actual_markers = (
            _make_hand_path_marker("/Visuals/HandReference/LeftActualPath", (0.0, 0.65, 1.0), 0.018),
            _make_hand_path_marker("/Visuals/HandReference/RightActualPath", (1.0, 0.50, 0.0), 0.018),
        )
        self.current_target_markers = (
            _make_hand_path_marker("/Visuals/HandReference/LeftCurrentTarget", (0.15, 1.0, 0.15), 0.040),
            _make_hand_path_marker("/Visuals/HandReference/RightCurrentTarget", (1.0, 0.15, 1.0), 0.040),
        )

        samples_per_control_step = max(1, int(round(self.env.step_dt / self.command_term.dataset_dt)))
        dataset_stride = self.stride * samples_per_control_step
        reference_path = self.command_term.reference_path_world(env_id=0, sample_stride=dataset_stride)
        if reference_path.shape[0] > self.max_points:
            extra_stride = math.ceil(reference_path.shape[0] / self.max_points)
            reference_path = reference_path[::extra_stride]
        self.reference_path = reference_path.detach()
        for hand_index in range(2):
            self.reference_markers[hand_index].visualize(self.reference_path[:, hand_index, :])

        print(
            "[INFO] Isaac Sim hand paths: "
            "left reference=green, right reference=magenta, "
            "left actual=cyan, right actual=orange; large spheres=current targets."
        )

    def update(self, step: int) -> None:
        state = self.command_term.tracking_state([0])
        torso_pos_w = self.robot.data.body_link_pos_w[0, self.torso_id, :]
        torso_quat_w = self.robot.data.body_link_quat_w[0, self.torso_id, :]
        torso_quat_hands = torso_quat_w.unsqueeze(0).expand(2, -1)

        target_pos_t = state["target_position"][0]
        actual_pos_t = state["actual_position"][0]
        target_pos_w = torso_pos_w.unsqueeze(0) + quat_apply(torso_quat_hands, target_pos_t)
        actual_pos_w = torso_pos_w.unsqueeze(0) + quat_apply(torso_quat_hands, actual_pos_t)

        for hand_index in range(2):
            self.current_target_markers[hand_index].visualize(target_pos_w[hand_index].unsqueeze(0))

        if step % self.stride != 0:
            return
        for hand_index in range(2):
            self.actual_paths[hand_index].append(actual_pos_w[hand_index].detach().clone())
            actual_path = torch.stack(tuple(self.actual_paths[hand_index]), dim=0)
            self.actual_markers[hand_index].visualize(actual_path)


class RslRlVecEnvWrapperExtraInfo(RslRlVecEnvWrapper):
    """Accept the lab-modified RSL-RL step signature while passing tensor actions to Isaac Lab."""

    def step(self, actions: torch.Tensor, extra_info=None):
        return super().step(actions)


def get_camera_position():
    """Get the current camera position from the USD stage.

    Returns:
        tuple: (x, y, z) camera position or None if not available
    """
    try:
        stage = stage_utils.get_current_stage()
        if stage is not None:
            # Get the viewport camera prim
            camera_prim_path = "/OmniverseKit_Persp"
            camera_prim = stage.GetPrimAtPath(camera_prim_path)

            if camera_prim and camera_prim.IsValid():
                # Get the camera's world transform
                camera_xform = UsdGeom.Xformable(camera_prim)
                world_transform = camera_xform.ComputeLocalToWorldTransform(0)  # 0 = current time

                # Extract position from the transform matrix
                camera_pos = world_transform.ExtractTranslation()
                return (camera_pos[0], camera_pos[1], camera_pos[2])
        return None
    except Exception as e:
        print(f"[ERROR]: Failed to get camera position: {e}")
        return None


def _configure_deterministic_evaluation(env_cfg) -> tuple[list[str], list[str]]:
    """Disable observation corruption and fix delayed actuators before environment creation."""
    if args_cli.eval_actuator_delay_steps < 0:
        raise ValueError("--eval_actuator_delay_steps must be non-negative.")

    disabled_observation_groups = []
    observation_cfg = getattr(env_cfg, "observations", None)
    if observation_cfg is not None:
        for group_name, group_cfg in vars(observation_cfg).items():
            if hasattr(group_cfg, "enable_corruption"):
                group_cfg.enable_corruption = False
                disabled_observation_groups.append(group_name)

    fixed_actuators = []
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    actuator_cfgs = getattr(robot_cfg, "actuators", {}) if robot_cfg is not None else {}
    for actuator_name, actuator_cfg in actuator_cfgs.items():
        if hasattr(actuator_cfg, "min_delay") and hasattr(actuator_cfg, "max_delay"):
            actuator_cfg.min_delay = args_cli.eval_actuator_delay_steps
            actuator_cfg.max_delay = args_cli.eval_actuator_delay_steps
            fixed_actuators.append(actuator_name)

    print(
        "[INFO] Deterministic evaluation configured: "
        f"observation_corruption_disabled={disabled_observation_groups}, "
        f"fixed_actuator_delay_steps={args_cli.eval_actuator_delay_steps}, "
        f"actuators={fixed_actuators}"
    )
    return disabled_observation_groups, fixed_actuators


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Honor CLI-provided overrides and defaults; avoid hard-coded debug presets
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if hasattr(env_cfg, "rebuild_dynamic_cfg"):
        env_cfg.rebuild_dynamic_cfg()

    if args_cli.hand_reference_csv is not None:
        if not hasattr(env_cfg.commands, "hand_reference"):
            raise ValueError("--hand_reference_csv requires a task with a hand_reference command.")
        reference_csv = os.path.abspath(os.path.expanduser(args_cli.hand_reference_csv))
        if not os.path.isfile(reference_csv):
            project_reference_csv = _project_path(args_cli.hand_reference_csv)
            if os.path.isfile(project_reference_csv):
                reference_csv = project_reference_csv
            else:
                raise FileNotFoundError(f"Hand-reference CSV does not exist: {reference_csv}")
        env_cfg.commands.hand_reference.data_path = reference_csv
        print(f"[INFO] Using hand-reference CSV override: {reference_csv}")

    # env_cfg.scene.env_spacing = 20.0
    
    
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = [-0.7, -0.7]
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = [0.0, 0.0]
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = [0.7, 0.7] # y正向左
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = [-1.0, 1.0]
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = [-1.5, 1.5] # 给0则yawKp失效
    # env_cfg.commands.base_velocity.ranges.heading = [1.57, 1.57]
    # S2 must keep its base command synchronized with the replay CSV (0.2 m/s).
    # Other tasks retain the local interactive-play command overrides.
    if not hasattr(env_cfg.commands, "hand_reference"):
        play_command_cfg = LOCAL_PLAY_DEFAULTS["command"]
        env_cfg.commands.base_velocity.ranges.lin_vel_x = play_command_cfg["lin_vel_x"]
        env_cfg.commands.base_velocity.ranges.lin_vel_y = play_command_cfg["lin_vel_y"]
        env_cfg.commands.base_velocity.ranges.ang_vel_z = play_command_cfg["ang_vel_z"]
        env_cfg.commands.base_velocity.heading_command = play_command_cfg["heading_command"]
        env_cfg.commands.base_velocity.rel_standing_envs = play_command_cfg["rel_standing_envs"]
    if hasattr(env_cfg.events, "reset_base"):
        reset_base_cfg = LOCAL_PLAY_DEFAULTS["reset_base"]
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = reset_base_cfg["yaw"]
        env_cfg.events.reset_base.params["pose_range"]["x"] = reset_base_cfg["x"]
        env_cfg.events.reset_base.params["pose_range"]["y"] = reset_base_cfg["y"]
    # env_cfg.viewer.eye = (2.0, 0.0, 2.0)
    # env_cfg.viewer.eye = (-1.0, 2.0, 2.0)
    # env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    # env_cfg.viewer.resolution = (1920,1080)
    # env_cfg.viewer.resolution = (1280,720)
    # env_cfg.viewer.origin_type = "asset_root"
    # env_cfg.viewer.asset_name = "robot"
    # env_cfg.viewer.env_index = 0

    # env_cfg.commands.base_velocity.debug_vis=False
    # env_cfg.commands.base_velocity.heading_control_stiffness = 1.0

    # 去除随机化
    # env_cfg.terminations = None
    # env_cfg.events = None
    env_cfg.curriculum = None    

    # set the environment seed
    # note: certain randomizations occur in environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    disabled_observation_groups = []
    fixed_actuators = []
    if args_cli.deterministic_eval:
        disabled_observation_groups, fixed_actuators = _configure_deterministic_evaluation(env_cfg)

    # GroundPlaneCfg references an online Isaac Sim grid USD.  On machines where that
    # asset is unavailable or still loading, its Plane child is missing and Isaac Lab
    # calls Stage.GetPrimAtPath(None).  A thin static cuboid provides the same flat
    # collision surface without any network/Nucleus dependency.  Keep this play-only
    # and opt-in so training configuration and existing grid visualization are unchanged.
    if args_cli.local_ground:
        if not hasattr(env_cfg.scene, "terrain"):
            raise ValueError("--local_ground requires env_cfg.scene.terrain.")
        env_cfg.scene.terrain.spawn = sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12)),
        )
        env_cfg.scene.terrain.init_state.pos = (0.0, 0.0, -0.025)
        print("[INFO] Using locally generated flat ground (remote grid USD disabled).")
        print("[WARNING] --local_ground differs from the grid ground used by the checkpoint's training run.")

    # specify directory for logging experiments
    # Prefer project-local logs under isaaclab_nhb/isaaclab_nhb/logs, then fallback to repo root logs
    script_dir = os.path.dirname(__file__)
    project_logs = os.path.abspath(os.path.join(script_dir, "..", "..", "logs", "rsl_rl", agent_cfg.experiment_name))
    repo_logs = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    # pick first existing; if neither exists, default to project_logs
    if os.path.isdir(project_logs):
        log_root_path = project_logs
    elif os.path.isdir(repo_logs):
        log_root_path = repo_logs
    else:
        log_root_path = project_logs
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            print("[INFO] Pre-trained checkpoint lookup is unavailable in this IsaacLab installation.")
            return
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        checkpoint_arg = os.path.expanduser(str(args_cli.checkpoint).strip().strip("'\""))
        if os.path.isfile(checkpoint_arg):
            resume_path = os.path.abspath(checkpoint_arg)
        else:
            resume_path = retrieve_file_path(checkpoint_arg)
    else:
        # find checkpoint under the chosen log root
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(
            "Resolved checkpoint path is not a file: "
            f"{resume_path!r} (requested: {args_cli.checkpoint!r})"
        )

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    wrapper_cls = RslRlVecEnvWrapperDictAction if hasattr(env.unwrapped, "action_extra_info") else RslRlVecEnvWrapperExtraInfo
    env = wrapper_cls(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if getattr(agent_cfg.policy, "class_name", None) == "ActorCriticResidual":
        # New S2 checkpoints contain the complete frozen S1 actor and both
        # normalizers, so playback must not depend on the original S1 file.
        agent_cfg.policy.defer_base_policy_load = True
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, load_optimizer=False)
    policy_module = runner.alg.policy
    if args_cli.diagnostic_csv:
        if hasattr(policy_module, "enable_action_component_recording"):
            policy_module.enable_action_component_recording(True)
            print("[INFO] Base/residual action decomposition recording enabled.")
        else:
            print("[WARNING] Policy does not expose base/residual action decomposition; those columns will be NaN.")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    
    # 使用日志文件夹名称作为onnx文件名,并添加模型步数
    log_folder_name = os.path.basename(log_dir)
    # 从checkpoint文件名中提取步数 (例如: model_18000.pt -> 18000)
    checkpoint_filename = os.path.basename(resume_path)
    model_step = checkpoint_filename.replace("model_", "").replace(".pt", "")
    onnx_filename = f"{log_folder_name}_step{model_step}.onnx"
    
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    try:
        export_policy_as_onnx(
            policy_nn,
            policy_type=agent_cfg.policy.class_name,
            normalizer=normalizer,
            path=export_model_dir,
            filename=onnx_filename,
        )
    except Exception as err:
        print(f"[WARNING] Skipping ONNX export during play: {err}")

    # 将obs和action到出为CSV
    if args_cli.csv_export:
        export_csv_dir = os.path.join(os.path.dirname(resume_path), "exported/csv")
        # 排除不需要记录的观测组：高程图和AMP观测
        datalogger = DataLogger(
            export_csv_dir, 
            env, 
            log_height_scan=False,
            exclude_obs_groups=['height_scan_policy', 'height_scan_critic', 'amp']
        )


    dt = env.unwrapped.step_dt
    hand_error_monitor = None
    if (
        args_cli.plot_hand_reference_errors
        or args_cli.plot_hand_reference_trajectory
        or args_cli.hand_error_csv
        or args_cli.diagnostic_csv
        or args_cli.hand_error_print_interval > 0
    ):
        try:
            diagnostic_metadata = {
                "task": args_cli.task,
                "checkpoint": resume_path,
                "device": str(env.unwrapped.device),
                "seed": agent_cfg.seed,
                "num_envs": env.unwrapped.num_envs,
                "deterministic_eval": args_cli.deterministic_eval,
                "observation_corruption_disabled": disabled_observation_groups,
                "fixed_actuator_delay_steps": (
                    args_cli.eval_actuator_delay_steps if args_cli.deterministic_eval else None
                ),
                "fixed_actuators": fixed_actuators,
                "ground_profile": "local_cuboid" if args_cli.local_ground else "task_default_training_grid",
                "hand_reference_csv": getattr(env.unwrapped.cfg.commands.hand_reference, "data_path", None),
            }
            hand_error_monitor = HandReferenceErrorMonitor(
                env=env,
                evidence_dir=_project_path("evidence"),
                enable_error_plot=args_cli.plot_hand_reference_errors,
                enable_trajectory_plot=args_cli.plot_hand_reference_trajectory,
                enable_csv=args_cli.hand_error_csv,
                enable_diagnostics=args_cli.diagnostic_csv,
                window=args_cli.hand_error_plot_window,
                print_interval=args_cli.hand_error_print_interval,
                metadata=diagnostic_metadata,
            )
        except RuntimeError as err:
            print(f"[WARNING] Hand-reference error monitor disabled: {err}")

    hand_reference_sim_viz = None
    if args_cli.show_hand_reference_in_sim:
        if args_cli.headless:
            print("[WARNING] --show_hand_reference_in_sim requires an Isaac Sim viewport; remove --headless.")
        try:
            hand_reference_sim_viz = HandReferenceInSimVisualizer(
                env,
                stride=args_cli.hand_reference_vis_stride,
                max_points=args_cli.hand_reference_vis_max_points,
            )
        except (KeyError, RuntimeError, ValueError) as err:
            print(f"[WARNING] Isaac Sim hand-reference visualization disabled: {err}")

    s1_hand_target_viz = None
    if args_cli.show_s1_hand_targets and not hasattr(env_cfg.commands, "hand_reference"):
        left_target_viz = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/S1LeftHandTarget",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    )
                },
            )
        )
        right_target_viz = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/S1RightHandTarget",
                markers={
                    "marker": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 1.0)),
                    )
                },
            )
        )
        s1_hand_target_viz = {
            "left_viz": left_target_viz,
            "right_viz": right_target_viz,
            "left_local": torch.tensor([[0.337, 0.309, 0.137]], device=env.unwrapped.device),
            "right_local": torch.tensor([[0.337, -0.309, 0.137]], device=env.unwrapped.device),
        }
        print("[INFO] Showing S1 hand target markers: red=left target, blue=right target.")

    # reset environment
    obs = env.get_observations()
    policy_feature_monitor = None
    if args_cli.plot_policy_features:
        policy_feature_monitor = PolicyFeatureMonitor(
            env=env,
            policy_module=policy_module,
            evidence_dir=_project_path("evidence"),
            metadata={
                "task": args_cli.task,
                "checkpoint": resume_path,
                "device": str(env.unwrapped.device),
                "seed": agent_cfg.seed,
                "num_envs": env.unwrapped.num_envs,
                "deterministic_eval": args_cli.deterministic_eval,
                "hand_reference_csv": getattr(
                    env.unwrapped.cfg.commands.hand_reference,
                    "data_path",
                    None,
                ),
            },
        )
    timestep = 0
    
    # determine if we need to track timesteps for limiting recording/playback
    timestep_limits = []
    if args_cli.video:
        timestep_limits.append(args_cli.video_length)
    if args_cli.csv_export:
        timestep_limits.append(args_cli.csv_length)
    if args_cli.play_steps is not None:
        if args_cli.play_steps <= 0:
            raise ValueError("--play_steps must be positive.")
        timestep_limits.append(args_cli.play_steps)
    should_limit_timesteps = bool(timestep_limits)
    max_timesteps = min(timestep_limits) if timestep_limits else None

    if should_limit_timesteps:
        print(
            f"[INFO] Play limited to {max_timesteps} timesteps "
            f"(video: {args_cli.video_length if args_cli.video else 'disabled'}, "
            f"csv: {args_cli.csv_length if args_cli.csv_export else 'disabled'}, "
            f"play: {args_cli.play_steps if args_cli.play_steps is not None else 'unlimited'})"
        )

    # Add a flag to skip camera position check in first few steps
    initial_steps = 5
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if policy_feature_monitor is not None:
                policy_feature_monitor.record(obs, timestep * dt)
            # agent stepping
            actions, extra_info = policy(obs)
            base_action = getattr(policy_module, "last_base_action", None)
            residual_action = getattr(policy_module, "last_residual_action", None)
            # env stepping
            obs, _, dones, step_extras = env.step(actions, extra_info)
            # The action-time estimate above belongs to obs_t. Diagnostics
            # below record the post-physics obs_{t+1}, so recompute only the
            # estimator head on that observation to keep force, frame, and
            # timestamp aligned.
            post_step_estimated_force = None
            if hand_error_monitor is not None:
                estimate_force = getattr(
                    policy_module, "estimate_virtual_force", None
                )
                if callable(estimate_force):
                    post_step_estimated_force = estimate_force(obs)

            done = bool(dones[0].detach().cpu())
            time_out = False
            if done and isinstance(step_extras, dict) and "time_outs" in step_extras:
                time_out_value = step_extras["time_outs"]
                if torch.is_tensor(time_out_value):
                    time_out = bool(time_out_value[0].detach().cpu())
                else:
                    time_out = bool(time_out_value)
            terminated = done and not time_out
            if policy_feature_monitor is not None:
                policy_feature_monitor.after_step(done)
            termination_reason = ""
            if done:
                active_reasons = [
                    name
                    for name, term_values in env.unwrapped.termination_manager.get_active_iterable_terms(0)
                    if term_values and float(term_values[0]) > 0.5
                ]
                termination_reason = "|".join(active_reasons) or ("time_out" if time_out else "unknown")

            if s1_hand_target_viz is not None:
                robot = env.unwrapped.scene["robot"]
                torso_id = robot.body_names.index("torso_link")
                torso_pos_w = robot.data.body_link_pos_w[:, torso_id, :]
                torso_quat_w = robot.data.body_link_quat_w[:, torso_id, :]

                left_target_w = torso_pos_w + quat_apply(
                    torso_quat_w,
                    s1_hand_target_viz["left_local"].expand(torso_pos_w.shape[0], -1),
                )
                right_target_w = torso_pos_w + quat_apply(
                    torso_quat_w,
                    s1_hand_target_viz["right_local"].expand(torso_pos_w.shape[0], -1),
                )
                s1_hand_target_viz["left_viz"].visualize(left_target_w)
                s1_hand_target_viz["right_viz"].visualize(right_target_w)

            if hand_error_monitor is not None:
                # The state above is post-physics, so its timestamp is the end
                # of this control step rather than the pre-step loop index.
                hand_error_monitor.update(
                    (timestep + 1) * dt,
                    done=done,
                    terminated=terminated,
                    time_out=time_out,
                    termination_reason=termination_reason,
                    total_action=actions,
                    base_action=base_action,
                    residual_action=residual_action,
                    estimated_force=post_step_estimated_force,
                )

            if hand_reference_sim_viz is not None:
                hand_reference_sim_viz.update(timestep)
        
        # Get and print camera position (skip first few steps to ensure physics is initialized)
        if timestep >= initial_steps:
            try:
                camera_pos = get_camera_position()
                if camera_pos is not None:
                    print(f"[Camera] Position: x={camera_pos[0]:.3f}, y={camera_pos[1]:.3f}, z={camera_pos[2]:.3f}")
            except Exception as e:
                # Silently skip if camera position retrieval fails
                pass
        
        # record CSV data
        if args_cli.csv_export:
            datalogger.log(obs, actions)

        # increment timestep and check if we should stop
        timestep += 1
        if should_limit_timesteps:
            if timestep >= max_timesteps:
                print(f"[INFO] Reached recording limit of {max_timesteps} timesteps, stopping...")
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if args_cli.csv_export:
        datalogger.close()
    if hand_error_monitor is not None:
        hand_error_monitor.close()
    if policy_feature_monitor is not None:
        policy_feature_monitor.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
