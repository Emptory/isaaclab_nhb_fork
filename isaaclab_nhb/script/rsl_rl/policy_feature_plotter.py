"""Export plots for the trajectory-related inputs consumed by the S2 actor."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


HANDS = ("left", "right")
AXES = ("x", "y", "z")
QUATERNION_COMPONENTS = ("w", "x", "y", "z")
ERROR_HISTORY_LAGS = (4, 3, 2, 1, 0)
ERROR_FIELDS = (
    ("position", "m", "position error [m]"),
    ("orientation", "rad", "orientation error [rad]"),
    ("linear_velocity", "mps", "linear velocity error [m/s]"),
    ("angular_velocity", "radps", "angular velocity error [rad/s]"),
)
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

REFERENCE_COLUMNS = tuple(
    column
    for hand in HANDS
    for column in (
        *(f"{hand}_reference_position_{axis}_m" for axis in AXES),
        *(f"{hand}_reference_quaternion_{component}" for component in QUATERNION_COMPONENTS),
        *(f"{hand}_reference_linear_velocity_{axis}_mps" for axis in AXES),
        *(f"{hand}_reference_angular_velocity_{axis}_radps" for axis in AXES),
    )
)
ERROR_HISTORY_COLUMNS = tuple(
    f"{hand}_{field}_error_lag{lag}_{axis}_{unit}"
    for lag in ERROR_HISTORY_LAGS
    for hand in HANDS
    for field, unit, _ in ERROR_FIELDS
    for axis in AXES
)
TARGET_FORCE_COLUMNS = tuple(
    f"{hand}_target_virtual_force_{axis}_n"
    for hand in HANDS
    for axis in AXES
)
FORCE_CONTROL_AXIS_COLUMNS = tuple(
    f"{hand}_force_control_axis_{axis}"
    for hand in HANDS
    for axis in AXES
)
GAIT_COLUMNS = (
    "gait_stance_rate",
    "gait_bipedal_offset",
    "gait_frequency_hz",
    "gait_sin_left",
    "gait_cos_left",
    "gait_sin_right",
    "gait_cos_right",
)
BASE_COMMAND_COLUMNS = (
    "base_command_x_mps",
    "base_command_y_mps",
    "base_command_yaw_radps",
)
ESTIMATED_FORCE_COLUMNS = tuple(
    f"{hand}_estimated_virtual_force_{axis}_n"
    for hand in HANDS
    for axis in AXES
)
APPENDED_BASE_ACTION_COLUMNS = tuple(
    f"appended_base_action_{joint_name}" for joint_name in ARM_JOINT_NAMES
)
SEMANTIC_COLUMNS = (
    *REFERENCE_COLUMNS,
    *ERROR_HISTORY_COLUMNS,
    *TARGET_FORCE_COLUMNS,
    *FORCE_CONTROL_AXIS_COLUMNS,
    *GAIT_COLUMNS,
    *BASE_COMMAND_COLUMNS,
    *ESTIMATED_FORCE_COLUMNS,
    *APPENDED_BASE_ACTION_COLUMNS,
)
IDENTITY_COLUMNS = ("step", "time_s", "episode_id", "episode_step")
MIN_Y_TICK_STEP = 0.01


def actor_input_columns(actor_input_dim: int) -> tuple[str, ...]:
    return tuple(f"actor_input_{index:03d}" for index in range(actor_input_dim))


def policy_feature_header(actor_input_dim: int) -> tuple[str, ...]:
    return (*IDENTITY_COLUMNS, *SEMANTIC_COLUMNS, *actor_input_columns(actor_input_dim))


def _read_csv(csv_path: str) -> dict[str, list[float]]:
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Policy-feature CSV has no header: {csv_path}")
        missing = [column for column in (*IDENTITY_COLUMNS, *SEMANTIC_COLUMNS) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Policy-feature CSV is missing columns: {missing}")
        data = {column: [] for column in reader.fieldnames}
        for row in reader:
            if not row:
                continue
            for column in data:
                value = row.get(column, "")
                data[column].append(float(value) if value not in (None, "") else float("nan"))
    if not data["time_s"]:
        raise ValueError(f"Policy-feature CSV contains no data rows: {csv_path}")
    return data


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
        roll_values.append(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
        pitch_values.append(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
        yaw_values.append(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll_values, pitch_values, yaw_values


def _apply_y_axis_scale(axis, minimum_tick_step: float = MIN_Y_TICK_STEP) -> None:
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
        axis.yaxis.set_major_locator(MultipleLocator(max(minimum_tick_step, nice_fraction * scale)))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{0.0 if abs(value) < 0.005 else value:.2f}")
    )


def _draw_reference_features(plt, data: dict[str, list[float]]):
    figure, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    time_s = data["time_s"]
    colors = ("tab:red", "tab:green", "tab:blue")
    for hand_index, hand in enumerate(HANDS):
        plot_specs = (
            (
                axes[0][hand_index],
                tuple(data[f"{hand}_reference_position_{axis}_m"] for axis in AXES),
                AXES,
                "reference position [m]",
            ),
            (
                axes[1][hand_index],
                _quaternion_series_to_rpy_rad(
                    *(data[f"{hand}_reference_quaternion_{component}"] for component in QUATERNION_COMPONENTS)
                ),
                ("roll", "pitch", "yaw"),
                "reference orientation [rad]",
            ),
            (
                axes[2][hand_index],
                tuple(data[f"{hand}_reference_linear_velocity_{axis}_mps"] for axis in AXES),
                AXES,
                "reference linear velocity [m/s]",
            ),
            (
                axes[3][hand_index],
                tuple(data[f"{hand}_reference_angular_velocity_{axis}_radps"] for axis in AXES),
                AXES,
                "reference angular velocity [rad/s]",
            ),
        )
        for row, (axis, series, labels, ylabel) in enumerate(plot_specs):
            for values, label, color in zip(series, labels, colors):
                axis.plot(time_s, values, label=label, color=color, linewidth=1.6)
            axis.set_title(f"{hand} hand in current torso frame")
            axis.set_ylabel(ylabel)
            axis.grid(True, linestyle="--", alpha=0.35)
            axis.legend(loc="best", fontsize=8)
            _apply_y_axis_scale(axis)
            if row == 3:
                axis.set_xlabel("play time [s]")
    figure.suptitle("Trajectory Reference Features Received by the S2 Policy")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return figure


def _error_norm_matrix(data, hand: str, field: str, unit: str, np):
    lag_rows = []
    for lag in ERROR_HISTORY_LAGS:
        components = np.asarray(
            [data[f"{hand}_{field}_error_lag{lag}_{axis}_{unit}"] for axis in AXES],
            dtype=float,
        )
        lag_rows.append(np.linalg.norm(components, axis=0))
    return np.asarray(lag_rows)


def _draw_error_history(plt, data: dict[str, list[float]]):
    import numpy as np

    figure, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True, sharey=True)
    time_s = np.asarray(data["time_s"], dtype=float)
    if len(time_s) == 1:
        time_extent = (time_s[0], time_s[0] + 1.0)
    else:
        time_extent = (time_s[0], time_s[-1])
    for field_index, (field, unit, label) in enumerate(ERROR_FIELDS):
        matrices = [_error_norm_matrix(data, hand, field, unit, np) for hand in HANDS]
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
        vmax = float(np.max(finite)) if finite.size else 1.0
        vmax = max(vmax, 1.0e-9)
        for hand_index, (hand, matrix) in enumerate(zip(HANDS, matrices)):
            axis = axes[field_index][hand_index]
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(*time_extent, -0.5, 4.5),
                vmin=0.0,
                vmax=vmax,
                cmap="viridis",
            )
            axis.set_title(f"{hand} hand: {label}")
            axis.set_yticks(range(5), ("t-4", "t-3", "t-2", "t-1", "t"))
            axis.set_ylabel("history frame")
            figure.colorbar(image, ax=axis, pad=0.01, label=label)
            if field_index == len(ERROR_FIELDS) - 1:
                axis.set_xlabel("play time [s]")
    figure.suptitle("Five-frame Tracking-error History Received by the S2 Policy")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return figure


def _plot_xyz(axis, time_s, data, prefix: str, suffix: str, ylabel: str) -> None:
    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    for component in AXES:
        column = f"{prefix}_{component}_{suffix}" if suffix else f"{prefix}_{component}"
        axis.plot(
            time_s,
            data[column],
            label=component,
            color=colors[component],
            linewidth=1.6,
        )
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best", fontsize=8)
    _apply_y_axis_scale(axis)


def _draw_context_features(plt, data: dict[str, list[float]]):
    import numpy as np

    figure, axes = plt.subplots(5, 2, figsize=(14, 16), sharex=True)
    time_s = data["time_s"]
    for hand_index, hand in enumerate(HANDS):
        _plot_xyz(
            axes[0][hand_index],
            time_s,
            data,
            f"{hand}_target_virtual_force",
            "n",
            "target force [N]",
        )
        axes[0][hand_index].set_title(f"{hand} target environment-on-hand force")
        _plot_xyz(
            axes[1][hand_index],
            time_s,
            data,
            f"{hand}_force_control_axis",
            "",
            "control-axis direction",
        )
        axes[1][hand_index].set_title(f"{hand} force-control axes")
        _plot_xyz(
            axes[2][hand_index],
            time_s,
            data,
            f"{hand}_estimated_virtual_force",
            "n",
            "estimated force [N]",
        )
        axes[2][hand_index].set_title(f"{hand} force estimate appended to actor input")

    for key, label in (
        ("gait_stance_rate", "stance rate"),
        ("gait_bipedal_offset", "left-right offset"),
        ("gait_frequency_hz", "frequency [Hz]"),
    ):
        axes[3][0].plot(time_s, data[key], label=label, linewidth=1.6)
    axes[3][0].set_title("Gait scalar commands")
    axes[3][0].set_ylabel("gait command")

    for key, label in (
        ("gait_sin_left", "sin left"),
        ("gait_cos_left", "cos left"),
        ("gait_sin_right", "sin right"),
        ("gait_cos_right", "cos right"),
    ):
        axes[3][1].plot(time_s, data[key], label=label, linewidth=1.4)
    axes[3][1].set_title("Gait phase channels")
    axes[3][1].set_ylabel("phase encoding")

    for key, label in zip(BASE_COMMAND_COLUMNS, ("vx", "vy", "yaw rate")):
        axes[4][0].plot(time_s, data[key], label=label, linewidth=1.6)
    axes[4][0].set_title("Base velocity command")
    axes[4][0].set_ylabel("m/s or rad/s")
    axes[4][0].set_xlabel("play time [s]")

    base_actions = np.asarray([data[column] for column in APPENDED_BASE_ACTION_COLUMNS], dtype=float)
    axes[4][1].plot(time_s, np.linalg.norm(base_actions, axis=0), label="L2 norm", linewidth=1.6)
    axes[4][1].plot(time_s, np.max(np.abs(base_actions), axis=0), label="max abs", linewidth=1.6)
    axes[4][1].set_title("Frozen-S1 arm action appended to actor input")
    axes[4][1].set_ylabel("action")
    axes[4][1].set_xlabel("play time [s]")

    for axis in axes.flat:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(loc="best", fontsize=8)
        _apply_y_axis_scale(axis)
    figure.suptitle("Command and Force Context Received by the S2 Policy")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return figure


def _write_csv(path: Path, values: dict[str, list[float]]) -> None:
    fields = list(values)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fields)
        writer.writerows(zip(*(values[field] for field in fields)))


def _write_plot_source_csvs(
    rawdata_dir: str,
    timestamp: str,
    data: dict[str, list[float]],
) -> list[str]:
    destination = Path(rawdata_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    identity_values = {column: data[column] for column in IDENTITY_COLUMNS}

    reference_values = {**identity_values, **{column: data[column] for column in REFERENCE_COLUMNS}}
    for hand in HANDS:
        rpy = _quaternion_series_to_rpy_rad(
            *(data[f"{hand}_reference_quaternion_{component}"] for component in QUATERNION_COMPONENTS)
        )
        for angle, values in zip(("roll", "pitch", "yaw"), rpy):
            reference_values[f"{hand}_reference_{angle}_rad"] = values

    error_values = {**identity_values, **{column: data[column] for column in ERROR_HISTORY_COLUMNS}}
    for lag in ERROR_HISTORY_LAGS:
        for hand in HANDS:
            for field, unit, _ in ERROR_FIELDS:
                components = zip(
                    *(data[f"{hand}_{field}_error_lag{lag}_{axis}_{unit}"] for axis in AXES)
                )
                error_values[f"{hand}_{field}_error_lag{lag}_norm_{unit}"] = [
                    math.sqrt(sum(value * value for value in row)) for row in components
                ]

    context_source_columns = (
        *TARGET_FORCE_COLUMNS,
        *FORCE_CONTROL_AXIS_COLUMNS,
        *GAIT_COLUMNS,
        *BASE_COMMAND_COLUMNS,
        *ESTIMATED_FORCE_COLUMNS,
        *APPENDED_BASE_ACTION_COLUMNS,
    )
    context_values = {**identity_values, **{column: data[column] for column in context_source_columns}}
    context_values["appended_base_action_l2"] = [
        math.sqrt(sum(value * value for value in row))
        for row in zip(*(data[column] for column in APPENDED_BASE_ACTION_COLUMNS))
    ]
    context_values["appended_base_action_max_abs"] = [
        max(abs(value) for value in row)
        for row in zip(*(data[column] for column in APPENDED_BASE_ACTION_COLUMNS))
    ]

    exports = {
        "policy_reference_features": reference_values,
        "policy_error_history": error_values,
        "policy_context_features": context_values,
    }
    outputs = []
    for stem, values in exports.items():
        output = destination / f"{stem}_{timestamp}.csv"
        _write_csv(output, values)
        outputs.append(str(output))
    return outputs


def save_policy_feature_plots(
    csv_path: str,
    output_dir: str,
    rawdata_dir: str,
    timestamp: str,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = _read_csv(csv_path)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    figures = {
        "policy_reference_features": _draw_reference_features(plt, data),
        "policy_error_history": _draw_error_history(plt, data),
        "policy_context_features": _draw_context_features(plt, data),
    }
    outputs = []
    for stem, figure in figures.items():
        output = destination / f"{stem}_{timestamp}.png"
        figure.savefig(output, dpi=200)
        plt.close(figure)
        outputs.append(str(output))
    for output in _write_plot_source_csvs(rawdata_dir, timestamp, data):
        print(f"[INFO] Policy-feature figure source data saved to: {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export S2 policy trajectory-feature plots.")
    parser.add_argument("--csv", required=True, help="Policy-feature CSV written by play.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for the three PNG files.")
    parser.add_argument("--rawdata-dir", required=True, help="Directory for per-figure source CSV files.")
    parser.add_argument("--timestamp", required=True, help="Timestamp suffix for exported files.")
    args = parser.parse_args()

    for output in save_policy_feature_plots(
        args.csv,
        args.output_dir,
        args.rawdata_dir,
        args.timestamp,
    ):
        print(f"[INFO] Policy-feature plot saved to: {output}")


if __name__ == "__main__":
    main()
