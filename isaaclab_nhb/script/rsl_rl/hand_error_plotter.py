import argparse
import csv
import math
import os
import time
from collections import deque


ERROR_KEYS = (
    "left_pos_error_m",
    "right_pos_error_m",
    "mean_pos_error_m",
    "mean_rot_error_deg",
    "mean_lin_vel_error_mps",
    "mean_ang_vel_error_radps",
    "mean_virtual_force_error_n",
    "mean_force_estimator_error_n",
)

MIN_Y_SPAN = 0.1


def _read_recent_rows(csv_path: str, window: int) -> dict[str, list[float]]:
    data = {"step": deque(maxlen=window)}
    for key in ERROR_KEYS:
        data[key] = deque(maxlen=window)

    if not os.path.exists(csv_path):
        return {key: list(value) for key, value in data.items()}

    try:
        with open(csv_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if not row:
                    continue
                data["step"].append(float(row["step"]))
                for key in ERROR_KEYS:
                    value = row.get(key)
                    data[key].append(
                        float(value) if value not in (None, "") else float("nan")
                    )
    except (OSError, ValueError, KeyError):
        pass

    return {key: list(value) for key, value in data.items()}


def _to_radians(values: list[float]) -> list[float]:
    return [math.radians(value) for value in values]


def _stable_upper_limit(
    previous_upper: float,
    *series: list[float],
    minimum_span: float = MIN_Y_SPAN,
) -> float:
    """Keep a zero-based axis at least ``minimum_span`` and never shrink it."""
    finite_values = [
        value
        for values in series
        for value in values
        if math.isfinite(value)
    ]
    observed_max = max(finite_values, default=0.0)
    required = max(minimum_span, 1.05 * observed_max)
    rounded = math.ceil(required / minimum_span) * minimum_span
    return max(previous_upper, rounded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live plot hand-reference tracking errors from a CSV file.")
    parser.add_argument("--csv", required=True, help="CSV file written by play.py --hand_error_csv.")
    parser.add_argument("--window", type=int, default=500, help="Number of recent steps to plot.")
    parser.add_argument("--refresh", type=float, default=0.2, help="Plot refresh period in seconds.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    plt.ion()
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    fig.canvas.manager.set_window_title("Hand Reference Tracking Errors")
    y_upper = [MIN_Y_SPAN] * len(axes)

    while plt.fignum_exists(fig.number):
        data = _read_recent_rows(args.csv, max(2, args.window))
        steps = data["step"]
        if steps:
            rotation_error_rad = _to_radians(data["mean_rot_error_deg"])
            for axis in axes:
                axis.clear()
                axis.grid(True)

            axes[0].plot(steps, data["left_pos_error_m"], label="left")
            axes[0].plot(steps, data["right_pos_error_m"], label="right")
            axes[0].plot(steps, data["mean_pos_error_m"], label="mean", linewidth=2)
            axes[0].set_ylabel("pos error [m]")
            axes[0].legend(loc="upper right")
            y_upper[0] = _stable_upper_limit(
                y_upper[0],
                data["left_pos_error_m"],
                data["right_pos_error_m"],
                data["mean_pos_error_m"],
            )
            axes[0].set_ylim(0.0, y_upper[0])

            axes[1].plot(steps, rotation_error_rad, color="tab:orange")
            axes[1].set_ylabel("rot error [rad]")
            y_upper[1] = _stable_upper_limit(y_upper[1], rotation_error_rad)
            axes[1].set_ylim(0.0, y_upper[1])

            axes[2].plot(steps, data["mean_lin_vel_error_mps"], color="tab:blue")
            axes[2].set_ylabel("lin vel error [m/s]")
            y_upper[2] = _stable_upper_limit(
                y_upper[2],
                data["mean_lin_vel_error_mps"],
            )
            axes[2].set_ylim(0.0, y_upper[2])

            axes[3].plot(steps, data["mean_ang_vel_error_radps"], color="tab:orange")
            axes[3].set_ylabel("ang vel error [rad/s]")
            y_upper[3] = _stable_upper_limit(
                y_upper[3],
                data["mean_ang_vel_error_radps"],
            )
            axes[3].set_ylim(0.0, y_upper[3])

            axes[4].plot(steps, data["mean_virtual_force_error_n"], label="tracking")
            axes[4].plot(steps, data["mean_force_estimator_error_n"], label="estimator")
            axes[4].set_ylabel("force error [N]")
            axes[4].set_xlabel("play step")
            axes[4].legend(loc="upper right")
            y_upper[4] = _stable_upper_limit(
                y_upper[4],
                data["mean_virtual_force_error_n"],
                data["mean_force_estimator_error_n"],
            )
            axes[4].set_ylim(0.0, y_upper[4])

            fig.tight_layout()
            fig.canvas.draw_idle()
        plt.pause(args.refresh)
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
