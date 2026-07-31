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
BASE_REQUIRED_COLUMNS = (
    "time_s",
    *POSITION_COLUMNS,
    *QUATERNION_COLUMNS,
)
PLOT_COLUMNS = (
    *BASE_REQUIRED_COLUMNS,
    *FORCE_COLUMNS,
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


def _quaternion_series_to_rpy_deg(
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
        roll_values.append(math.degrees(math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)))
        pitch_values.append(math.degrees(math.asin(sin_pitch)))
        yaw_values.append(math.degrees(math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)))
    return roll_values, pitch_values, yaw_values


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
            series[source] = _quaternion_series_to_rpy_deg(*quaternion)

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
            axis.set_ylabel(f"{angle_name} [deg]")
            axis.grid(True, linestyle="--", alpha=0.35)
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


def _draw_force_components(axes, data: dict[str, list[float]]) -> None:
    """Draw environment-on-hand target, applied spring, and estimated forces."""
    time_s = data["time_s"]
    styles = {
        "target": ("k--", 2.0, "target env-on-hand"),
        "actual": ("-", 1.7, "actual virtual spring"),
        "estimated": (":", 1.7, "actor estimate"),
    }
    colors = {"actual": "tab:red", "estimated": "tab:green"}
    for hand_index, hand in enumerate(HANDS):
        for axis_index, axis_name in enumerate(AXES):
            axis = axes[axis_index][hand_index]
            axis.clear()
            for source in ("target", "actual", "estimated"):
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


def _create_figures(plt):
    position_fig, position_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    position_fig.canvas.manager.set_window_title("Hand Reference vs Actual: World Position")
    orientation_fig, orientation_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    orientation_fig.canvas.manager.set_window_title("Hand Reference vs Actual: World Orientation")
    path_fig = plt.figure(figsize=(12, 5))
    path_fig.canvas.manager.set_window_title("Hand Reference vs Actual: 3D Paths")
    path_axes = [path_fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    force_fig, force_axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    force_fig.canvas.manager.set_window_title(
        "Virtual Force: Target vs Applied vs Estimated"
    )
    return (
        position_fig,
        position_axes,
        orientation_fig,
        orientation_axes,
        path_fig,
        path_axes,
        force_fig,
        force_axes,
    )


def _draw_all(figures, data: dict[str, list[float]]) -> None:
    (
        position_fig,
        position_axes,
        orientation_fig,
        orientation_axes,
        path_fig,
        path_axes,
        force_fig,
        force_axes,
    ) = figures
    _draw_position_components(position_axes, data)
    _draw_orientation_components(orientation_axes, data)
    _draw_3d_paths(path_axes, data)
    _draw_force_components(force_axes, data)
    position_fig.tight_layout()
    orientation_fig.tight_layout()
    path_fig.tight_layout()
    force_fig.tight_layout()


def _save_plots(csv_path: str, output_prefix: str | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = _read_recent_rows(csv_path, 10_000_000)
    if not data["time_s"]:
        raise ValueError(f"Tracking CSV has no complete trajectory rows: {csv_path}")
    figures = _create_figures(plt)
    _draw_all(figures, data)
    prefix = Path(output_prefix).expanduser() if output_prefix else Path(csv_path).with_suffix("")
    outputs = [
        str(prefix) + "_position.png",
        str(prefix) + "_orientation.png",
        str(prefix) + "_path_3d.png",
        str(prefix) + "_virtual_force.png",
    ]
    for figure, output in zip(
        (figures[0], figures[2], figures[4], figures[6]), outputs
    ):
        figure.savefig(output, dpi=200)
        plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aligned hand references against actual hand states.")
    parser.add_argument("--csv", required=True, help="Tracking CSV written by play.py.")
    parser.add_argument("--window", type=int, default=500, help="Number of recent play steps shown live.")
    parser.add_argument("--refresh", type=float, default=0.2, help="Plot refresh period in seconds.")
    parser.add_argument("--save-only", action="store_true", help="Write PNGs and exit without opening windows.")
    parser.add_argument("--output-prefix", default=None, help="Prefix for --save-only PNG paths.")
    args = parser.parse_args()

    if args.save_only:
        for output in _save_plots(args.csv, args.output_prefix):
            print(f"[INFO] Hand tracking plot saved to: {output}")
        return

    import matplotlib.pyplot as plt

    plt.ion()
    figures = _create_figures(plt)
    figure_objects = (figures[0], figures[2], figures[4], figures[6])
    while any(plt.fignum_exists(figure.number) for figure in figure_objects):
        data = _read_recent_rows(args.csv, max(2, args.window))
        if data["time_s"]:
            _draw_all(figures, data)
            for figure in figure_objects:
                figure.canvas.draw_idle()
        plt.pause(args.refresh)


if __name__ == "__main__":
    main()
