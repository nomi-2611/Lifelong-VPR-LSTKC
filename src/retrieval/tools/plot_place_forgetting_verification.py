from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY = {
    "nordland_place": "Nordland",
    "robotcar_place": "RobotCar",
    "tart_place": "Tart",
}


def parse_stage_results(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage_blocks = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|Average" in line and "place" in line:
            names = [part.strip() for part in line.split("|") if part.strip() and "Average" not in part]
            if names:
                stage = len(names)
                if stage not in stage_blocks:
                    values_line = lines[i + 1] if i + 1 < len(lines) else ""
                    values = [part.strip() for part in values_line.split("|") if part.strip()]
                    result = {}
                    for name, value in zip(names, values):
                        m, r = value.split("/")
                        result[name] = {"mAP": float(m), "R1": float(r)}
                    stage_blocks[stage] = result
        i += 1
    return stage_blocks


def plot_forgetting(af_on, af_off, out_path: Path):
    stages = [1, 2, 3]
    stage_labels = ["After T1", "After T2", "After T3"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=260, constrained_layout=True)
    colors = {"nordland_place": "#1f77b4", "robotcar_place": "#2ca02c"}

    for metric, ax in zip(["mAP", "R1"], axes):
        for dataset in ["nordland_place", "robotcar_place"]:
            on_x, on_y, off_x, off_y = [], [], [], []
            for stage in stages:
                if dataset in af_on.get(stage, {}):
                    on_x.append(stage)
                    on_y.append(af_on[stage][dataset][metric])
                if dataset in af_off.get(stage, {}):
                    off_x.append(stage)
                    off_y.append(af_off[stage][dataset][metric])

            ax.plot(on_x, on_y, marker="o", linewidth=2.4, color=colors[dataset], label=f"{DISPLAY[dataset]} (AF on)")
            ax.plot(off_x, off_y, marker="o", linewidth=2.0, linestyle="--", color=colors[dataset], alpha=0.9,
                    label=f"{DISPLAY[dataset]} (AF off)")

        ax.set_xticks(stages)
        ax.set_xticklabels(stage_labels)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(f"{metric} (%)")
        ax.grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.01, "Fast sequential setting: 256 place IDs per stage, 1 epoch per stage.", ha="center", fontsize=10, color="#475569")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot place forgetting verification figure.")
    parser.add_argument("--af-on-log", required=True)
    parser.add_argument("--af-off-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    af_on = parse_stage_results(Path(args.af_on_log))
    af_off = parse_stage_results(Path(args.af_off_log))
    plot_forgetting(af_on, af_off, Path(args.output))


if __name__ == "__main__":
    main()
