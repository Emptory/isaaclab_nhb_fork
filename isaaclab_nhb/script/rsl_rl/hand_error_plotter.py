import argparse
import csv
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
)


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
                    data[key].append(float(row[key]))
    except (OSError, ValueError, KeyError):
        pass

    return {key: list(value) for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live plot hand-reference tracking errors from a CSV file.")
    parser.add_argument("--csv", required=True, help="CSV file written by play.py --hand_error_csv.")
    parser.add_argument("--window", type=int, default=500, help="Number of recent steps to plot.")
    parser.add_argument("--refresh", type=float, default=0.2, help="Plot refresh period in seconds.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.canvas.manager.set_window_title("Hand Reference Tracking Errors")

    while plt.fignum_exists(fig.number):
        data = _read_recent_rows(args.csv, max(2, args.window))
        steps = data["step"]
        if steps:
            for axis in axes:
                axis.clear()
                axis.grid(True)

            axes[0].plot(steps, data["left_pos_error_m"], label="left")
            axes[0].plot(steps, data["right_pos_error_m"], label="right")
            axes[0].plot(steps, data["mean_pos_error_m"], label="mean", linewidth=2)
            axes[0].set_ylabel("pos error [m]")
            axes[0].legend(loc="upper right")

            axes[1].plot(steps, data["mean_rot_error_deg"], color="tab:orange")
            axes[1].set_ylabel("rot error [deg]")

            axes[2].plot(steps, data["mean_lin_vel_error_mps"], label="lin [m/s]")
            axes[2].plot(steps, data["mean_ang_vel_error_radps"], label="ang [rad/s]")
            axes[2].set_ylabel("vel error")
            axes[2].set_xlabel("play step")
            axes[2].legend(loc="upper right")

            fig.tight_layout()
            fig.canvas.draw_idle()
        plt.pause(args.refresh)
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
