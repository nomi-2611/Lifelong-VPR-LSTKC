from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["nordland_place", "robotcar_place", "tart_place", "Average"]
DISPLAY_NAMES = {
    "nordland_place": "Nordland",
    "robotcar_place": "RobotCar",
    "tart_place": "Tart",
    "Average": "Avg",
}


def short_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def parse_log_res(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    values = [item for item in lines[1].split("|") if item.strip()]
    parsed = {}
    names = ["nordland_place", "robotcar_place", "tart_place", "Average"]
    for name, item in zip(names, values):
        m, r = item.strip().split("/")
        parsed[name] = {"mAP": float(m), "R1": float(r)}
    return parsed


def load_summary(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        parsed.append({k: float(v) for k, v in row.items()})
    parsed.sort(key=lambda x: x["weight"])
    return parsed


def plot_method_comparison(matrix_res, embedding_res, fusion_res, out_path: Path):
    methods = ["Matrix", "LSTKC Feature", "Fusion"]
    colors = ["#c95f44", "#4f7bd9", "#1f9d72"]
    display_datasets = [short_name(name) for name in DATASETS]
    x = np.arange(len(DATASETS))
    width = 0.23

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=260, constrained_layout=True)

    for idx, (name, res) in enumerate(zip(methods, [matrix_res, embedding_res, fusion_res])):
        mvals = [res["nordland_place"]["mAP"], res["robotcar_place"]["mAP"], res["tart_place"]["mAP"], res["Average"]["mAP"]]
        axes[0].bar(x + (idx - 1) * width, mvals, width=width, label=name, color=colors[idx])
    axes[0].set_title("mAP", fontweight="bold")
    axes[0].set_ylabel("mAP (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_datasets)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    for idx, (name, res) in enumerate(zip(methods, [matrix_res, embedding_res, fusion_res])):
        rvals = [res["nordland_place"]["R1"], res["robotcar_place"]["R1"], res["tart_place"]["R1"], res["Average"]["R1"]]
        axes[1].bar(x + (idx - 1) * width, rvals, width=width, label=name, color=colors[idx])
    axes[1].set_title("R@1", fontweight="bold")
    axes[1].set_ylabel("R@1 (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_datasets)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fusion_curve(summary_rows, out_path: Path):
    xs = [row["weight"] for row in summary_rows]
    avg_map = [row["average_mAP"] for row in summary_rows]
    avg_r1 = [row["average_R1"] for row in summary_rows]
    robot_map = [row["robotcar_place_mAP"] for row in summary_rows]
    robot_r1 = [row["robotcar_place_R1"] for row in summary_rows]

    best_avg_idx = max(range(len(summary_rows)), key=lambda i: (avg_map[i], avg_r1[i]))
    best_robot_idx = max(range(len(summary_rows)), key=lambda i: (robot_map[i], robot_r1[i]))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), dpi=260, constrained_layout=True)

    axes[0].plot(xs, avg_map, marker="o", linewidth=2.5, color="#1f77b4", label="Average mAP")
    axes[0].plot(xs, avg_r1, marker="s", linewidth=2.5, color="#ff7f0e", label="Average R@1")
    axes[0].scatter([xs[best_avg_idx]], [avg_map[best_avg_idx]], color="#1f77b4", s=50, zorder=5)
    axes[0].annotate(
        f"Best avg\n{xs[best_avg_idx]:.2f}: {avg_map[best_avg_idx]:.1f}/{avg_r1[best_avg_idx]:.1f}",
        xy=(xs[best_avg_idx], avg_map[best_avg_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f77b4"),
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
    )
    axes[0].set_title("Average vs Weight", fontweight="bold")
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_xticks(xs)
    avg_all = avg_map + avg_r1
    axes[0].set_ylim(min(avg_all) - 0.8, max(avg_all) + 0.6)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(xs, robot_map, marker="o", linewidth=2.5, color="#2ca02c", label="robotcar mAP")
    axes[1].plot(xs, robot_r1, marker="s", linewidth=2.5, color="#9467bd", label="robotcar R@1")
    axes[1].scatter([xs[best_robot_idx]], [robot_map[best_robot_idx]], color="#2ca02c", s=50, zorder=5)
    axes[1].annotate(
        f"Best RobotCar\n{xs[best_robot_idx]:.2f}: {robot_map[best_robot_idx]:.1f}/{robot_r1[best_robot_idx]:.1f}",
        xy=(xs[best_robot_idx], robot_map[best_robot_idx]),
        xytext=(10, -38),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2ca02c"),
        arrowprops=dict(arrowstyle="->", color="#2ca02c"),
    )
    axes[1].set_title("RobotCar vs Weight", fontweight="bold")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_xticks(xs)
    robot_all = robot_map + robot_r1
    axes[1].set_ylim(min(robot_all) - 1.0, max(robot_all) + 0.8)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_dataset_curve(summary_rows, out_path: Path):
    xs = [row["weight"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=260, constrained_layout=True)
    ax.plot(xs, [row["nordland_place_mAP"] for row in summary_rows], marker="o", linewidth=2.4, color="#d95f02", label="Nordland")
    ax.plot(xs, [row["robotcar_place_mAP"] for row in summary_rows], marker="s", linewidth=2.4, color="#1f77b4", label="RobotCar")
    ax.plot(xs, [row["tart_place_mAP"] for row in summary_rows], marker="^", linewidth=2.4, color="#7570b3", label="Tart")
    ax.plot(xs, [row["average_mAP"] for row in summary_rows], marker="D", linewidth=2.4, color="#2ca02c", label="Avg")
    ax.set_title("Per-dataset mAP", fontweight="bold")
    ax.set_xlabel("Weight")
    ax.set_ylabel("mAP (%)")
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate VPR-only result figures.")
    parser.add_argument("--matrix-log", required=True)
    parser.add_argument("--embedding-log", required=True)
    parser.add_argument("--fusion-log", required=True)
    parser.add_argument("--fusion-summary-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    matrix_res = parse_log_res(Path(args.matrix_log))
    embedding_res = parse_log_res(Path(args.embedding_log))
    fusion_res = parse_log_res(Path(args.fusion_log))
    summary_rows = load_summary(Path(args.fusion_summary_csv))

    plot_method_comparison(matrix_res, embedding_res, fusion_res, out_dir / "vpr_method_comparison.png")
    plot_fusion_curve(summary_rows, out_dir / "vpr_fusion_weight_curve.png")
    plot_per_dataset_curve(summary_rows, out_dir / "vpr_per_dataset_map_curve.png")


if __name__ == "__main__":
    main()
