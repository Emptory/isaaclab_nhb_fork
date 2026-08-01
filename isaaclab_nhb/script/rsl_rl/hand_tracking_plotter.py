"""Live world-frame plots of aligned hand references and measured hand-anchor states."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque
from pathlib import Path


HANDS = ("left", "right")
AXES = ("x", "y", "z")
MIN_Y_TICK_STEP = 0.01
POSITION_COLUMNS = tuple(
    f"{hand}_{source}_position_{axis}_m"
    for hand in HANDS
    for source in ("target", "actual")
    for axis in AXES
)
QUATERNION_COLUMNS = tuple(
    f"{hand}_{source}_quaternion_{component}"
    for hand in HANDS
    for source in ("target", "actual")
    for component in ("w", "x", "y", "z")
)
FORCE_COLUMNS = tuple(
    f"{hand}_{source}_force_{axis}_n"
    for hand in HANDS
    for source in ("target", "actual", "estimated")
    for axis in AXES
)
LINEAR_VELOCITY_COLUMNS = tuple(
    f"{hand}_{source}_linear_velocity_{axis}_mps"
    for hand in HANDS
    for source in ("target", "actual")
    for axis in AXES
)
ANGULAR_VELOCITY_COLUMNS = tuple(
    f"{hand}_{source}_angular_velocity_{axis}_radps"
    for hand in HANDS
    for source in ("target", "actual")
    for axis in AXES
)
ERROR_COLUMNS = (
    "left_pos_error_m",
    "right_pos_error_m",
    "mean_pos_error_m",
    "mean_rot_error_deg",
    "mean_lin_vel_error_mps",
    "mean_ang_vel_error_radps",
)
HAND_ERROR_COMPONENT_COLUMNS = tuple(
    f"{hand}_{field}_error_world_{axis}_{unit}"
    for hand in HANDS
    for field, unit in (
        ("position", "m"),
        ("rotation", "deg"),
        ("linear_velocity", "mps"),
        ("angular_velocity", "radps"),
    )
    for axis in AXES
)
GAIT_DIAGNOSTIC_COLUMNS = (
    "torso_rpy_world_roll_deg",
    "torso_rpy_world_pitch_deg",
    "torso_angular_velocity_world_x_radps",
    "torso_angular_velocity_world_y_radps",
    "torso_height_world_m",
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
BASE_REQUIRED_COLUMNS = (
    "time_s",
    *POSITION_COLUMNS,
    *QUATERNION_COLUMNS,
)
PLOT_COLUMNS = (
    *BASE_REQUIRED_COLUMNS,
    *LINEAR_VELOCITY_COLUMNS,
    *ANGULAR_VELOCITY_COLUMNS,
    *FORCE_COLUMNS,
    *ERROR_COLUMNS,
    *HAND_ERROR_COMPONENT_COLUMNS,
    *GAIT_DIAGNOSTIC_COLUMNS,
)


def _read_recent_rows(csv_path: str, window: int) -> dict[str, list[float]]:
    data = {key: deque(maxlen=window) for key in PLOT_COLUMNS}
    if not os.path.exists(csv_path):
        return {key: [] for key in PLOT_COLUMNS}

    try:
        with open(csv_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or any(
                key not in reader.fieldnames for key in BASE_REQUIRED_COLUMNS
            ):
                return {key: [] for key in PLOT_COLUMNS}
            available_columns = set(reader.fieldnames)
            for row in reader:
                if not row:
                    continue
                for key in PLOT_COLUMNS:
                    value = row[key] if key in available_columns else "nan"
                    data[key].append(float(value))
    except (OSError, ValueError, KeyError):
        pass
    return {key: list(value) for key, value in data.items()}


def _quaternion_series_to_rpy_rad(
    w_values: list[float],
    x_values: list[float],
    y_values: list[float],
    z_values: list[float],
) -> tuple[list[float], list[float], list[float]]:
    roll_values = []
    pitch_values = []
    yaw_values = []
    for w, x, y, z in zip(w_values, x_values, y_values, z_values):
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1.0e-12:
            roll_values.append(float("nan"))
            pitch_values.append(float("nan"))
            yaw_values.append(float("nan"))
            continue
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        sin_roll_cos_pitch = 2.0 * (w * x + y * z)
        cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
        sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
        cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
        roll_values.append(math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch))
        pitch_values.append(math.asin(sin_pitch))
        yaw_values.append(math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch))
    return roll_values, pitch_values, yaw_values


def _apply_y_axis_scale(axis, minimum_tick_step: float = MIN_Y_TICK_STEP) -> None:
    """Keep small variations readable without allowing sub-centimetre/sub-0.01-rad ticks."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    finite_values = [
        float(value)
        for line in axis.lines
        for value in line.get_ydata()
        if math.isfinite(float(value))
    ]
    if finite_values:
        data_span = max(finite_values) - min(finite_values)
        raw_step = max(minimum_tick_step, data_span / 8.0)
        exponent = math.floor(math.log10(raw_step))
        scale = 10.0**exponent
        fraction = raw_step / scale
        nice_fraction = next(
            candidate for candidate in (1.0, 2.0, 2.5, 5.0, 10.0) if candidate >= fraction
        )
        tick_step = max(minimum_tick_step, nice_fraction * scale)
        axis.yaxis.set_major_locator(MultipleLocator(tick_step))

    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{0.0 if abs(value) < 0.005 else value:.2f}")
    )


def _draw_position_components(axes, data: dict[str, list[float]]) -> None:
    time_s = data["time_s"]
    for hand_index, hand in enumerate(HANDS):
        for axis_index, axis_name in enumerate(AXES):
            axis = axes[axis_index][hand_index]
            axis.clear()
            axis.plot(
                time_s,
                data[f"{hand}_target_position_{axis_name}_m"],
                "k--",
                linewidth=2.0,
                label="reference (offline CSV, aligned)",
            )
            axis.plot(
                time_s,
                data[f"{hand}_actual_position_{axis_name}_m"],
                color="tab:blue",
                linewidth=1.7,
                label="actual palm anchor",
            )
            axis.set_title(f"{hand} hand: {axis_name}")
            axis.set_ylabel(f"{axis_name} [m]")
            axis.grid(True, linestyle="--", alpha=0.35)
            _apply_y_axis_scale(axis)
            if axis_index == 0:
                axis.legend(loc="best", fontsize=8)
            if axis_index == len(AXES) - 1:
                axis.set_xlabel("play time [s]")


def _draw_velocity_components(
    axes,
    data: dict[str, list[float]],
    field: str,
    unit: str,
) -> None:
    time_s = data["time_s"]
    for hand_index, hand in enumerate(HANDS):
        for axis_index, axis_name in enumerate(AXES):
            axis = axes[axis_index][hand_index]
            axis.clear()
            axis.plot(
                time_s,
                data[f"{hand}_target_{field}_{axis_name}_{unit}"],
                "k--",
                linewidth=2.0,
                label="reference (offline CSV, aligned)",
            )
            axis.plot(
                time_s,
                data[f"{hand}_actual_{field}_{axis_name}_{unit}"],
                color="tab:purple",
                linewidth=1.7,
                label="actual hand",
            )
            axis.set_title(f"{hand} hand: {axis_name}")
            axis.set_ylabel(f"{axis_name} [{unit}]")
            axis.grid(True, linestyle="--", alpha=0.35)
            _apply_y_axis_scale(axis)
            if axis_index == 0:
                axis.legend(loc="best", fontsize=8)
            if axis_index == len(AXES) - 1:
                axis.set_xlabel("play time [s]")


def _draw_orientation_components(axes, data: dict[str, list[float]]) -> None:
    time_s = data["time_s"]
    angle_names = ("roll", "pitch", "yaw")
    for hand_index, hand in enumerate(HANDS):
        series = {}
        for source in ("target", "actual"):
            quaternion = tuple(data[f"{hand}_{source}_quaternion_{component}"] for component in ("w", "x", "y", "z"))
            series[source] = _quaternion_series_to_rpy_rad(*quaternion)

        for angle_index, angle_name in enumerate(angle_names):
            axis = axes[angle_index][hand_index]
            axis.clear()
            axis.plot(
                time_s,
                series["target"][angle_index],
                "k--",
                linewidth=2.0,
                label="reference (offline CSV, aligned)",
            )
            axis.plot(
                time_s,
                series["actual"][angle_index],
                color="tab:orange",
                linewidth=1.7,
                label="actual hand",
            )
            axis.set_title(f"{hand} hand: {angle_name}")
            axis.set_ylabel(f"{angle_name} [rad]")
            axis.grid(True, linestyle="--", alpha=0.35)
            _apply_y_axis_scale(axis)
            if angle_index == 0:
                axis.legend(loc="best", fontsize=8)
            if angle_index == len(angle_names) - 1:
                axis.set_xlabel("play time [s]")


def _draw_3d_paths(axes, data: dict[str, list[float]]) -> None:
    for hand_index, hand in enumerate(HANDS):
        axis = axes[hand_index]
        axis.clear()
        axis.plot(
            data[f"{hand}_target_position_x_m"],
            data[f"{hand}_target_position_y_m"],
            data[f"{hand}_target_position_z_m"],
            "k--",
            linewidth=2.0,
            label="reference (aligned)",
        )
        axis.plot(
            data[f"{hand}_actual_position_x_m"],
            data[f"{hand}_actual_position_y_m"],
            data[f"{hand}_actual_position_z_m"],
            color="tab:blue",
            linewidth=1.7,
            label="actual",
        )
        axis.set_title(f"{hand} hand path in world frame")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend(loc="best", fontsize=8)


def _draw_force_components(
    axes,
    data: dict[str, list[float]],
    include_estimated_force: bool = False,
) -> None:
    """Draw environment-on-hand target, applied spring, and estimated forces."""
    time_s = data["time_s"]
    styles = {
        "target": ("k--", 2.0, "target env-on-hand"),
        "actual": ("-", 1.7, "actual virtual spring"),
        "estimated": (":", 1.7, "actor estimate"),
    }
    colors = {"actual": "tab:red", "estimated": "tab:green"}
    sources = ("target", "actual", "estimated") if include_estimated_force else ("target", "actual")
    for hand_index, hand in enumerate(HANDS):
        for axis_index, axis_name in enumerate(AXES):
            axis = axes[axis_index][hand_index]
            axis.clear()
            for source in sources:
                linestyle, linewidth, label = styles[source]
                kwargs = {}
                if source in colors:
                    kwargs["color"] = colors[source]
                axis.plot(
                    time_s,
                    data[f"{hand}_{source}_force_{axis_name}_n"],
                    linestyle,
                    linewidth=linewidth,
                    label=label,
                    **kwargs,
                )
            axis.set_title(f"{hand} hand force: {axis_name}")
            axis.set_ylabel(f"F{axis_name} [N]")
            axis.grid(True, linestyle="--", alpha=0.35)
            if axis_index == 0:
                axis.legend(loc="best", fontsize=8)
            if axis_index == len(AXES) - 1:
                axis.set_xlabel("play time [s]")


def _draw_tracking_errors(axes, data: dict[str, list[float]]) -> None:
    time_s = data["time_s"]
    series = (
        ("position error [m]", ("left_pos_error_m", "right_pos_error_m"), ("left", "right")),
        ("orientation error [rad]", ("mean_rot_error_deg",), ("mean",)),
        ("linear velocity error [m/s]", ("mean_lin_vel_error_mps",), ("mean",)),
        ("angular velocity error [rad/s]", ("mean_ang_vel_error_radps",), ("mean",)),
    )
    for axis, (ylabel, keys, labels) in zip(axes, series):
        axis.clear()
        for key, label in zip(keys, labels):
            values = data[key]
            if key == "mean_rot_error_deg":
                values = [
                    math.radians(value) if math.isfinite(value) else float("nan")
                    for value in values
                ]
            axis.plot(time_s, values, linewidth=1.7, label=label)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.35)
        _apply_y_axis_scale(axis)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("play time [s]")


def _draw_tracking_error_components(axes, data: dict[str, list[float]]) -> None:
    time_s = data["time_s"]
    fields = (
        ("position", "m", "position error [m]"),
        ("rotation", "deg", "rotation error [rad]"),
        ("linear_velocity", "mps", "linear velocity error [m/s]"),
        ("angular_velocity", "radps", "angular velocity error [rad/s]"),
    )
    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    for field_index, (field, source_unit, ylabel) in enumerate(fields):
        for hand_index, hand in enumerate(HANDS):
            axis = axes[field_index][hand_index]
            axis.clear()
            for axis_name in AXES:
                values = data[f"{hand}_{field}_error_world_{axis_name}_{source_unit}"]
                if field == "rotation":
                    values = [
                        math.radians(value) if math.isfinite(value) else float("nan")
                        for value in values
                    ]
                axis.plot(time_s, values, color=colors[axis_name], label=axis_name)
            axis.set_title(f"{hand} hand")
            axis.set_ylabel(ylabel)
            axis.grid(True, linestyle="--", alpha=0.35)
            _apply_y_axis_scale(axis)
            axis.legend(loc="best", fontsize=8)
            if field_index == len(fields) - 1:
                axis.set_xlabel("play time [s]")


def _draw_torso_gait(axes, data: dict[str, list[float]]) -> None:
    time_s = data["time_s"]
    axes[0].clear()
    axes[0].plot(
        time_s,
        [
            math.radians(value) if math.isfinite(value) else float("nan")
            for value in data["torso_rpy_world_roll_deg"]
        ],
        label="torso roll",
    )
    axes[0].plot(
        time_s,
        [
            math.radians(value) if math.isfinite(value) else float("nan")
            for value in data["torso_rpy_world_pitch_deg"]
        ],
        label="torso pitch",
    )
    axes[0].set_ylabel("torso angle [rad]")

    axes[1].clear()
    axes[1].plot(time_s, data["torso_angular_velocity_world_x_radps"], label="torso wx")
    axes[1].plot(time_s, data["torso_angular_velocity_world_y_radps"], label="torso wy")
    axes[1].set_ylabel("torso angular velocity [rad/s]")

    axes[2].clear()
    axes[2].plot(time_s, data["left_pos_error_m"], label="left hand")
    axes[2].plot(time_s, data["right_pos_error_m"], label="right hand")
    axes[2].set_ylabel("position error [m]")

    axes[3].clear()
    axes[3].plot(time_s, data["gait_phase"], label="gait phase")
    axes[3].plot(time_s, data["gait_frequency_hz"], linestyle="--", label="gait frequency [Hz]")
    axes[3].plot(time_s, data["gait_sin"], linestyle=":", label="gait sin")
    axes[3].plot(time_s, data["gait_cos"], linestyle=":", label="gait cos")
    axes[3].set_ylabel("gait")
    axes[3].set_ylim(-1.05, 1.05)

    axes[4].clear()
    axes[4].plot(time_s, data["base_command_x_mps"], label="command vx")
    axes[4].plot(time_s, data["base_command_y_mps"], label="command vy")
    axes[4].plot(time_s, data["base_command_yaw_radps"], label="command yaw")
    axes[4].set_ylabel("base command")
    axes[4].set_xlabel("play time [s]")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(loc="best", fontsize=8)
        _apply_y_axis_scale(axis)


def _has_error_data(data: dict[str, list[float]]) -> bool:
    return any(
        math.isfinite(value)
        for key in ERROR_COLUMNS
        for value in data.get(key, [])
    )


def _has_gait_data(data: dict[str, list[float]]) -> bool:
    return any(
        math.isfinite(value)
        for key in GAIT_DIAGNOSTIC_COLUMNS
        for value in data.get(key, [])
    )


def _write_derived_csvs(rawdata_dir: str, timestamp: str, data: dict[str, list[float]]) -> list[str]:
    """Write one raw CSV per exported figure while retaining the full tracking CSV."""
    os.makedirs(rawdata_dir, exist_ok=True)
    time_s = data["time_s"]
    exports: dict[str, dict[str, list[float]]] = {}

    exports["position_xyz"] = {
        "time_s": time_s,
        **{key: data[key] for key in POSITION_COLUMNS},
    }
    orientation_values = {"time_s": time_s}
    for hand in HANDS:
        for source in ("target", "actual"):
            quaternion = tuple(
                data[f"{hand}_{source}_quaternion_{component}"]
                for component in ("w", "x", "y", "z")
            )
            roll, pitch, yaw = _quaternion_series_to_rpy_rad(*quaternion)
            for name, values in zip(("roll", "pitch", "yaw"), (roll, pitch, yaw)):
                orientation_values[f"{hand}_{source}_{name}_rad"] = values
    exports["orientation_rpy"] = orientation_values
    exports["linear_velocity_xyz"] = {"time_s": time_s, **{key: data[key] for key in LINEAR_VELOCITY_COLUMNS}}
    exports["angular_velocity_xyz"] = {"time_s": time_s, **{key: data[key] for key in ANGULAR_VELOCITY_COLUMNS}}
    exports["force_xyz"] = {
        "time_s": time_s,
        **{key: data[key] for key in FORCE_COLUMNS if "_estimated_" not in key},
    }

    error_values = {"time_s": time_s}
    for key in ("left_pos_error_m", "right_pos_error_m", "mean_pos_error_m", "mean_lin_vel_error_mps", "mean_ang_vel_error_radps"):
        error_values[key] = data[key]
    error_values["mean_rot_error_rad"] = [
        math.radians(value) if math.isfinite(value) else float("nan")
        for value in data["mean_rot_error_deg"]
    ]
    for key in HAND_ERROR_COMPONENT_COLUMNS:
        if "_rotation_error_" in key:
            output_key = key.removesuffix("_deg") + "_rad"
            error_values[output_key] = [
                math.radians(value) if math.isfinite(value) else float("nan")
                for value in data[key]
            ]
        else:
            error_values[key] = data[key]
    exports["tracking_error"] = error_values

    torso_values = {"time_s": time_s}
    for source, target in (
        ("torso_rpy_world_roll_deg", "torso_roll_rad"),
        ("torso_rpy_world_pitch_deg", "torso_pitch_rad"),
    ):
        torso_values[target] = [
            math.radians(value) if math.isfinite(value) else float("nan")
            for value in data[source]
        ]
    for key in (
        "torso_angular_velocity_world_x_radps",
        "torso_angular_velocity_world_y_radps",
        "torso_height_world_m",
        "left_pos_error_m",
        "right_pos_error_m",
        "gait_frequency_hz",
        "gait_phase",
        "gait_sin",
        "gait_cos",
        "gait_sin_right",
        "gait_cos_right",
        "base_command_x_mps",
        "base_command_y_mps",
        "base_command_yaw_radps",
    ):
        torso_values[key] = data[key]
    exports["torso_gait"] = torso_values

    outputs = []
    for stem, values in exports.items():
        output = Path(rawdata_dir) / f"{stem}_{timestamp}.csv"
        fields = list(values)
        with open(output, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fields)
            writer.writerows(zip(*(values[field] for field in fields)))
        outputs.append(str(output))
    return outputs


def _create_figures(
    plt,
    include_3d_path: bool = False,
    include_error: bool = False,
    include_torso_gait: bool = False,
):
    position_fig, position_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    position_fig.canvas.manager.set_window_title("Hand Reference vs Actual: World Position")
    orientation_fig, orientation_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    orientation_fig.canvas.manager.set_window_title("Hand Reference vs Actual: World Orientation")
    force_fig, force_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    force_fig.canvas.manager.set_window_title(
        "Virtual Force: Target vs Actual"
    )
    figures = {
        "position": (position_fig, position_axes),
        "orientation": (orientation_fig, orientation_axes),
        "force": (force_fig, force_axes),
    }
    linear_velocity_fig, linear_velocity_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    linear_velocity_fig.canvas.manager.set_window_title("Hand Reference vs Actual: Linear Velocity")
    figures["linear_velocity"] = (linear_velocity_fig, linear_velocity_axes)
    angular_velocity_fig, angular_velocity_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    angular_velocity_fig.canvas.manager.set_window_title("Hand Reference vs Actual: Angular Velocity")
    figures["angular_velocity"] = (angular_velocity_fig, angular_velocity_axes)
    if include_error:
        error_fig, error_axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        error_fig.canvas.manager.set_window_title("Hand Reference Tracking Errors")
        figures["tracking_error"] = (error_fig, error_axes)
        component_fig, component_axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
        component_fig.canvas.manager.set_window_title("Per-axis Hand Tracking Errors")
        figures["tracking_error_components"] = (component_fig, component_axes)
    if include_torso_gait:
        torso_fig, torso_axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
        torso_fig.canvas.manager.set_window_title("Torso Sway and Gait Phase")
        figures["torso_gait"] = (torso_fig, torso_axes)
    if include_3d_path:
        path_fig = plt.figure(figsize=(12, 5))
        path_fig.canvas.manager.set_window_title("Hand Reference vs Actual: 3D Paths")
        path_axes = [path_fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
        figures["path_3d"] = (path_fig, path_axes)
    return figures


def _draw_all(
    figures,
    data: dict[str, list[float]],
    include_estimated_force: bool = False,
) -> None:
    position_fig, position_axes = figures["position"]
    orientation_fig, orientation_axes = figures["orientation"]
    force_fig, force_axes = figures["force"]
    _draw_position_components(position_axes, data)
    _draw_orientation_components(orientation_axes, data)
    linear_velocity_fig, linear_velocity_axes = figures["linear_velocity"]
    _draw_velocity_components(linear_velocity_axes, data, "linear_velocity", "mps")
    angular_velocity_fig, angular_velocity_axes = figures["angular_velocity"]
    _draw_velocity_components(angular_velocity_axes, data, "angular_velocity", "radps")
    _draw_force_components(force_axes, data, include_estimated_force)
    position_fig.tight_layout()
    orientation_fig.tight_layout()
    linear_velocity_fig.tight_layout()
    angular_velocity_fig.tight_layout()
    force_fig.tight_layout()
    if "tracking_error" in figures:
        error_fig, error_axes = figures["tracking_error"]
        _draw_tracking_errors(error_axes, data)
        error_fig.tight_layout()
        component_fig, component_axes = figures["tracking_error_components"]
        _draw_tracking_error_components(component_axes, data)
        component_fig.tight_layout()
    if "torso_gait" in figures:
        torso_fig, torso_axes = figures["torso_gait"]
        _draw_torso_gait(torso_axes, data)
        torso_fig.tight_layout()
    if "path_3d" in figures:
        path_fig, path_axes = figures["path_3d"]
        _draw_3d_paths(path_axes, data)
        path_fig.tight_layout()


def _save_plots(
    csv_path: str,
    output_prefix: str | None,
    output_dir: str | None = None,
    rawdata_dir: str | None = None,
    timestamp: str | None = None,
    include_3d_path: bool = False,
    include_estimated_force: bool = False,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = _read_recent_rows(csv_path, 10_000_000)
    if not data["time_s"]:
        raise ValueError(f"Tracking CSV has no complete trajectory rows: {csv_path}")
    figures = _create_figures(
        plt,
        include_3d_path=include_3d_path,
        include_error=_has_error_data(data),
        include_torso_gait=_has_gait_data(data),
    )
    _draw_all(figures, data, include_estimated_force=include_estimated_force)
    if output_dir:
        destination = Path(output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        name_timestamp = timestamp or Path(csv_path).stem
        outputs_by_name = {
            "position": destination / f"position_xyz_{name_timestamp}.png",
            "orientation": destination / f"orientation_rpy_{name_timestamp}.png",
            "linear_velocity": destination / f"linear_velocity_xyz_{name_timestamp}.png",
            "angular_velocity": destination / f"angular_velocity_xyz_{name_timestamp}.png",
            "force": destination / f"force_xyz_{name_timestamp}.png",
            "tracking_error": destination / f"tracking_error_{name_timestamp}.png",
            "tracking_error_components": destination / f"tracking_error_components_{name_timestamp}.png",
            "torso_gait": destination / f"torso_gait_{name_timestamp}.png",
            "path_3d": destination / f"path_3d_{name_timestamp}.png",
        }
    else:
        prefix = Path(output_prefix).expanduser() if output_prefix else Path(csv_path).with_suffix("")
        outputs_by_name = {
            "position": Path(str(prefix) + "_position.png"),
            "orientation": Path(str(prefix) + "_orientation.png"),
            "linear_velocity": Path(str(prefix) + "_linear_velocity.png"),
            "angular_velocity": Path(str(prefix) + "_angular_velocity.png"),
            "force": Path(str(prefix) + "_virtual_force.png"),
            "tracking_error": Path(str(prefix) + "_tracking_error.png"),
            "tracking_error_components": Path(str(prefix) + "_tracking_error_components.png"),
            "torso_gait": Path(str(prefix) + "_torso_gait.png"),
            "path_3d": Path(str(prefix) + "_path_3d.png"),
        }
    outputs = []
    for name, (figure, _) in figures.items():
        output = str(outputs_by_name[name])
        figure.savefig(output, dpi=200)
        plt.close(figure)
        outputs.append(output)
    if rawdata_dir:
        raw_timestamp = timestamp or Path(csv_path).stem
        for output in _write_derived_csvs(rawdata_dir, raw_timestamp, data):
            print(f"[INFO] Figure source data saved to: {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aligned hand references against actual hand states.")
    parser.add_argument("--csv", required=True, help="Tracking CSV written by play.py.")
    parser.add_argument("--window", type=int, default=500, help="Number of recent play steps shown live.")
    parser.add_argument("--refresh", type=float, default=0.2, help="Plot refresh period in seconds.")
    parser.add_argument("--save-only", action="store_true", help="Write PNGs and exit without opening windows.")
    parser.add_argument("--output-prefix", default=None, help="Prefix for --save-only PNG paths.")
    parser.add_argument("--output-dir", default=None, help="Directory for clearly named --save-only PNGs.")
    parser.add_argument("--rawdata-dir", default=None, help="Directory for one source CSV per exported figure.")
    parser.add_argument("--timestamp", default=None, help="Timestamp suffix used together with --output-dir.")
    parser.add_argument("--include-3d-path", action="store_true", help="Also show/export a 3D path figure.")
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Show live tracking-error and torso/gait figures (save-only detects available data automatically).",
    )
    parser.add_argument(
        "--include-estimated-force",
        action="store_true",
        help="Also draw the actor force estimate in the force figure.",
    )
    args = parser.parse_args()

    if args.save_only:
        for output in _save_plots(
            args.csv,
            args.output_prefix,
            output_dir=args.output_dir,
            rawdata_dir=args.rawdata_dir,
            timestamp=args.timestamp,
            include_3d_path=args.include_3d_path,
            include_estimated_force=args.include_estimated_force,
        ):
            print(f"[INFO] Hand tracking plot saved to: {output}")
        return

    import matplotlib.pyplot as plt

    plt.ion()
    figures = _create_figures(
        plt,
        include_3d_path=args.include_3d_path,
        include_error=args.include_diagnostics,
        include_torso_gait=args.include_diagnostics,
    )
    figure_objects = tuple(figure for figure, _ in figures.values())
    while any(plt.fignum_exists(figure.number) for figure in figure_objects):
        data = _read_recent_rows(args.csv, max(2, args.window))
        if data["time_s"]:
            _draw_all(figures, data, include_estimated_force=args.include_estimated_force)
            for figure in figure_objects:
                figure.canvas.draw_idle()
        plt.pause(args.refresh)


if __name__ == "__main__":
    main()
