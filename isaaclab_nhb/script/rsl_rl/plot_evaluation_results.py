"""Plot report figures from evaluate_checkpoints.py CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


parser = argparse.ArgumentParser(description="Generate report plots from evaluation CSV files.")
parser.add_argument("--summary_csv", required=True, help="Path to eval_summary.csv.")
parser.add_argument("--output_dir", default=None, help="Directory for generated figures and tables.")
parser.add_argument("--trajectory_csv", action="append", default=None, help="Optional trajectory CSV to plot.")
args = parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _group_by_policy(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["policy_name"], []).append(row)
    return grouped


def _write_markdown_table(rows: list[dict[str, str]], output_path: Path) -> None:
    columns = [
        ("Policy", "policy_name"),
        ("v_cmd (m/s)", "v_cmd_x"),
        ("Success Rate", "success_rate"),
        ("Speed RMSE (m/s)", "speed_rmse_xy"),
        ("Speed Err Mean (m/s)", "speed_error_mean_xy"),
        ("Hand Err (m)", "hand_pos_error_mean"),
        ("Torso AngVel", "torso_ang_vel_xy_mean"),
        ("Pelvis AngVel", "pelvis_ang_vel_xy_mean"),
        ("Action Rate", "action_rate_mean"),
    ]
    with open(output_path, "w") as file:
        file.write("| " + " | ".join(title for title, _ in columns) + " |\n")
        file.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            values = []
            for _, key in columns:
                value = row.get(key, "")
                numeric = _to_float(value)
                if math.isfinite(numeric):
                    if key == "success_rate":
                        values.append(f"{numeric * 100:.1f}%")
                    elif key == "v_cmd_x":
                        values.append(f"{numeric:.2f}")
                    else:
                        values.append(f"{numeric:.4f}")
                else:
                    values.append(value)
            file.write("| " + " | ".join(values) + " |\n")


def _plot_metric_by_velocity(rows: list[dict[str, str]], metric: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _group_by_policy(rows)
    plt.figure(figsize=(7.2, 4.2))
    for policy_name, policy_rows in grouped.items():
        policy_rows = sorted(policy_rows, key=lambda row: _to_float(row["v_cmd_x"]))
        xs = [_to_float(row["v_cmd_x"]) for row in policy_rows]
        ys = [_to_float(row[metric]) for row in policy_rows]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=policy_name)
    plt.xlabel("Command velocity $v_x$ (m/s)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_average_metric(rows: list[dict[str, str]], metric: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _group_by_policy(rows)
    names = []
    values = []
    for policy_name, policy_rows in grouped.items():
        metric_values = [_to_float(row[metric]) for row in policy_rows]
        metric_values = [value for value in metric_values if math.isfinite(value)]
        if metric_values:
            names.append(policy_name)
            values.append(sum(metric_values) / len(metric_values))
    plt.figure(figsize=(7.2, 4.0))
    plt.bar(names, values, width=0.55)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_trajectory(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_rows(path)
    if not rows:
        return
    time_s = [_to_float(row["time_s"]) for row in rows]
    command_vx = [_to_float(row["command_vx"]) for row in rows]
    actual_vx = [_to_float(row["actual_vx"]) for row in rows]
    speed_error = [_to_float(row["speed_error_xy"]) for row in rows]

    stem = path.stem
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(time_s, command_vx, linewidth=2.0, label="$v_{cmd,x}$")
    plt.plot(time_s, actual_vx, linewidth=1.8, label="$v_x$")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_velocity_tracking.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 3.5))
    plt.plot(time_s, speed_error, linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed error (m/s)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_speed_error.png", dpi=300)
    plt.close()


def main() -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as err:
        raise SystemExit(
            "matplotlib is not installed in the current Python environment. "
            "Run this script inside the IsaacLab/conda environment."
        ) from err

    summary_csv = Path(args.summary_csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else summary_csv.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(summary_csv)

    _write_markdown_table(rows, output_dir / "eval_summary_table.md")
    _plot_metric_by_velocity(rows, "speed_rmse_xy", "Speed RMSE (m/s)", output_dir / "speed_rmse_by_velocity.png")
    _plot_metric_by_velocity(rows, "success_rate", "Success rate", output_dir / "success_rate_by_velocity.png")
    _plot_metric_by_velocity(rows, "hand_pos_error_mean", "Hand position error (m)", output_dir / "hand_error_by_velocity.png")
    _plot_average_metric(rows, "speed_rmse_xy", "Mean speed RMSE (m/s)", output_dir / "mean_speed_rmse_ablation.png")
    _plot_average_metric(rows, "success_rate", "Mean success rate", output_dir / "mean_success_rate_ablation.png")

    if args.trajectory_csv:
        for trajectory_csv in args.trajectory_csv:
            _plot_trajectory(Path(trajectory_csv).resolve(), output_dir)

    print(f"[INFO] Figures and table saved to: {output_dir}")


if __name__ == "__main__":
    main()
