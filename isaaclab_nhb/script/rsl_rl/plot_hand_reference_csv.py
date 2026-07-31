"""Plot and summarize the raw two-hand offline reference CSV.

This script does not launch Isaac Sim. It visualizes the dataset-frame values
before the reset-time alignment performed by ``TwoHandCsvReferenceCommand``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_CSV = Path(__file__).resolve().parents[2] / "stimulationData" / "box_6dof_two_hand_19x2.csv"
HANDS = ("left", "right")
XYZ = ("x", "y", "z")
QUAT = ("qw", "qx", "qy", "qz")
ANGULAR = ("wx", "wy", "wz")
FORCE = ("Fx", "Fy", "Fz")
MOMENT = ("Mx", "My", "Mz")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a two-hand 19x2 offline reference CSV.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Two-hand reference CSV path.")
    parser.add_argument("--output", default=None, help="Output PNG path (default: next to the CSV).")
    parser.add_argument("--show", action="store_true", help="Also open an interactive Matplotlib window.")
    return parser.parse_args()


def _load_csv(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames or []
        required = ["t"]
        for hand in HANDS:
            required.extend(f"{hand}_{name}" for name in (*XYZ, *QUAT, "vx", "vy", "vz", *ANGULAR, *FORCE, *MOMENT))
        missing = [name for name in required if name not in headers]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        data = {name: [] for name in required}
        for row in reader:
            for name in required:
                data[name].append(float(row[name]))

    if len(data["t"]) < 2:
        raise ValueError("Reference CSV must contain at least two samples.")
    return headers, data


def _print_summary(path: Path, data: dict[str, list[float]]) -> None:
    times = data["t"]
    dt = (times[-1] - times[0]) / (len(times) - 1)
    print(f"[INFO] CSV: {path}")
    print(f"[INFO] samples={len(times)}, duration={times[-1] - times[0]:.6f} s, dt={dt:.6f} s")
    for hand in HANDS:
        ranges = []
        for axis in XYZ:
            values = data[f"{hand}_{axis}"]
            ranges.append(f"{axis}=[{min(values):.6f}, {max(values):.6f}]")
        print(f"[INFO] {hand} position (dataset frame): " + ", ".join(ranges))
        velocity = tuple(data[f"{hand}_v{axis}"][0] for axis in XYZ)
        force = tuple(data[f"{hand}_F{axis}"][0] for axis in XYZ)
        print(f"[INFO] {hand} initial linear velocity={velocity} m/s, force={force} N")


def _plot(data: dict[str, list[float]], output: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    time_s = data["t"]
    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    hand_styles = {"left": "-", "right": "--"}

    fig = plt.figure(figsize=(14, 15))
    grid = fig.add_gridspec(4, 2)
    axes = {
        "path": fig.add_subplot(grid[0, 0], projection="3d"),
        "x": fig.add_subplot(grid[0, 1]),
        "y": fig.add_subplot(grid[1, 0]),
        "z": fig.add_subplot(grid[1, 1]),
        "quat": fig.add_subplot(grid[2, 0]),
        "lin_vel": fig.add_subplot(grid[2, 1]),
        "ang_vel": fig.add_subplot(grid[3, 0]),
        "wrench": fig.add_subplot(grid[3, 1]),
    }

    for hand in HANDS:
        axes["path"].plot(
            data[f"{hand}_x"], data[f"{hand}_y"], data[f"{hand}_z"],
            hand_styles[hand], linewidth=2, label=hand,
        )
        for axis in XYZ:
            axes[axis].plot(
                time_s, data[f"{hand}_{axis}"], hand_styles[hand],
                color=colors[axis], label=hand,
            )

    axes["path"].set_title("Raw two-hand position paths (dataset frame N)")
    axes["path"].set_xlabel("x [m]")
    axes["path"].set_ylabel("y [m]")
    axes["path"].set_zlabel("z [m]")
    axes["path"].legend()
    for axis in XYZ:
        axes[axis].set_title(f"{axis} position")
        axes[axis].set_xlabel("time [s]")
        axes[axis].set_ylabel(f"{axis} [m]")
        axes[axis].legend()

    quat_colors = dict(zip(QUAT, ("black", "tab:red", "tab:green", "tab:blue")))
    for hand in HANDS:
        for component in QUAT:
            axes["quat"].plot(
                time_s, data[f"{hand}_{component}"], hand_styles[hand],
                color=quat_colors[component], label=f"{hand} {component}",
            )
    axes["quat"].set_title("Quaternion (scalar first, body B to dataset frame N)")
    axes["quat"].set_xlabel("time [s]")
    axes["quat"].set_ylabel("quaternion component")
    axes["quat"].legend(ncol=2, fontsize=8)

    for hand in HANDS:
        for axis in XYZ:
            axes["lin_vel"].plot(
                time_s, data[f"{hand}_v{axis}"], hand_styles[hand],
                color=colors[axis], label=f"{hand} v{axis}",
            )
            axes["ang_vel"].plot(
                time_s, data[f"{hand}_w{axis}"], hand_styles[hand],
                color=colors[axis], label=f"{hand} w{axis}",
            )
    axes["lin_vel"].set_title("Linear velocity in dataset frame N")
    axes["lin_vel"].set_xlabel("time [s]")
    axes["lin_vel"].set_ylabel("velocity [m/s]")
    axes["lin_vel"].legend(ncol=2, fontsize=8)
    axes["ang_vel"].set_title("Angular velocity in payload body frame B")
    axes["ang_vel"].set_xlabel("time [s]")
    axes["ang_vel"].set_ylabel("angular velocity [rad/s]")
    axes["ang_vel"].legend(ncol=2, fontsize=8)

    for hand in HANDS:
        for axis in XYZ:
            axes["wrench"].plot(
                time_s, data[f"{hand}_F{axis}"], hand_styles[hand],
                color=colors[axis], label=f"{hand} F{axis}",
            )
    axes["wrench"].set_title("Offline contact-force command")
    axes["wrench"].set_xlabel("time [s]")
    axes["wrench"].set_ylabel("force [N]")
    axes["wrench"].legend(ncol=2, fontsize=8)

    for name, axis in axes.items():
        if name != "path":
            axis.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("Two-hand offline reference (raw CSV, before play-time alignment)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"[INFO] Offline reference plot saved to: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Reference CSV does not exist: {csv_path}")
    output = Path(args.output).expanduser().resolve() if args.output else csv_path.with_name(f"{csv_path.stem}_overview.png")
    _, data = _load_csv(csv_path)
    _print_summary(csv_path, data)
    _plot(data, output, args.show)


if __name__ == "__main__":
    main()
