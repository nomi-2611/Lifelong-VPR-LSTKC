import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_stage_results(log_res_path):
    lines = [line.strip() for line in Path(log_res_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    stages = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("epoch:"):
            i += 1
            continue

        if i + 5 >= len(lines) or lines[i + 1].startswith("epoch:"):
            i += 1
            continue

        epoch = int(lines[i].split(":", 1)[1].strip())
        seen_names = [name for name in lines[i + 1].split("\t") if name and name != "|Average"]
        seen_vals = [float(x) for x in re.findall(r"\d+\.\d+", lines[i + 2])]
        seen = {}
        for idx, name in enumerate(seen_names):
            seen[name] = {"mAP": seen_vals[idx * 2], "R1": seen_vals[idx * 2 + 1]}
        seen_avg = {"mAP": seen_vals[-2], "R1": seen_vals[-1]}

        unseen_names = [name for name in lines[i + 4].split("\t") if name and name != "|Average"]
        unseen_vals = [float(x) for x in re.findall(r"\d+\.\d+", lines[i + 5])]
        unseen = {}
        for idx, name in enumerate(unseen_names):
            unseen[name] = {"mAP": unseen_vals[idx * 2], "R1": unseen_vals[idx * 2 + 1]}
        unseen_avg = {"mAP": unseen_vals[-2], "R1": unseen_vals[-1]}

        stages.append(
            {
                "epoch": epoch,
                "seen": seen,
                "seen_avg": seen_avg,
                "unseen": unseen,
                "unseen_avg": unseen_avg,
            }
        )
        i += 6
    return stages


def load_tensorboard_scalars(run_dir):
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    series = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        series[tag] = {
            "steps": np.array([e.step for e in events], dtype=float),
            "values": np.array([e.value for e in events], dtype=float),
        }
    return series


def style():
    available = set(plt.style.available)
    if "seaborn-v0_8-whitegrid" in available:
        plt.style.use("seaborn-v0_8-whitegrid")
    elif "seaborn-whitegrid" in available:
        plt.style.use("seaborn-whitegrid")
    else:
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "legend.frameon": True,
        }
    )


def plot_stage_summary(stages, out_path):
    final_stage = stages[-1]
    datasets = list(final_stage["seen"].keys())
    maps = [final_stage["seen"][d]["mAP"] for d in datasets]
    r1s = [final_stage["seen"][d]["R1"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, maps, width, label="mAP", color="#1f77b4")
    ax.bar(x + width / 2, r1s, width, label="R1", color="#ff7f0e")
    ax.axhline(final_stage["seen_avg"]["mAP"], color="#1f77b4", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(final_stage["seen_avg"]["R1"], color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title("Final Stage Performance by Dataset")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    note = f"Seen average: {final_stage['seen_avg']['mAP']:.1f} mAP / {final_stage['seen_avg']['R1']:.1f} R1"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=10, color="#444444")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_stage_progression(stages, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    stage_labels = [f"Stage {idx + 1}" for idx in range(len(stages))]
    seen_map = [stage["seen_avg"]["mAP"] for stage in stages]
    seen_r1 = [stage["seen_avg"]["R1"] for stage in stages]
    unseen_map = [stage["unseen_avg"]["mAP"] for stage in stages]
    unseen_r1 = [stage["unseen_avg"]["R1"] for stage in stages]

    axes[0].plot(stage_labels, seen_map, marker="o", linewidth=2.5, color="#1f77b4", label="Seen mAP")
    axes[0].plot(stage_labels, seen_r1, marker="o", linewidth=2.5, color="#ff7f0e", label="Seen R1")
    axes[0].set_title("Seen Performance Across Stages")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="best")

    axes[1].plot(stage_labels, unseen_map, marker="o", linewidth=2.5, color="#2ca02c", label="Unseen mAP")
    axes[1].plot(stage_labels, unseen_r1, marker="o", linewidth=2.5, color="#d62728", label="Unseen R1")
    axes[1].set_title("Unseen Performance Across Stages")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_retention(stages, out_path):
    if len(stages) < 2:
        return

    prev_stage = stages[-2]
    final_stage = stages[-1]
    shared = [name for name in prev_stage["seen"] if name in final_stage["seen"]]

    map_delta = [final_stage["seen"][name]["mAP"] - prev_stage["seen"][name]["mAP"] for name in shared]
    r1_delta = [final_stage["seen"][name]["R1"] - prev_stage["seen"][name]["R1"] for name in shared]

    x = np.arange(len(shared))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, map_delta, width, label="mAP delta", color="#4c78a8")
    ax.bar(x + width / 2, r1_delta, width, label="R1 delta", color="#f58518")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Retention After Adding the Final Stage")
    ax.set_ylabel("Stage 3 minus Stage 2 (points)")
    ax.set_xticks(x)
    ax.set_xticklabels(shared)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(series, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
    colors = {"ce": "#1f77b4", "tr": "#d62728"}

    for stage_idx in range(1, 4):
        ce_tag = f"loss/Loss_ce_{stage_idx}"
        tr_tag = f"loss/Loss_tr_{stage_idx}"
        ax = axes[stage_idx - 1]

        if ce_tag in series:
            ax.plot(series[ce_tag]["steps"], series[ce_tag]["values"], color=colors["ce"], linewidth=1.6, label="CE loss")
        if tr_tag in series:
            ax.plot(series[tr_tag]["steps"], series[tr_tag]["values"], color=colors["tr"], linewidth=1.6, label="Triplet loss")

        ax.set_title(f"Stage {stage_idx} Training Loss")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("TensorBoard step")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate training analysis plots from an LSTKC run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to a run directory such as logs/run_full_3stage")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    style()
    stages = parse_stage_results(run_dir / "log_res.txt")
    series = load_tensorboard_scalars(run_dir)

    plot_stage_summary(stages, out_dir / "stage_summary.png")
    plot_stage_progression(stages, out_dir / "stage_progression.png")
    plot_retention(stages, out_dir / "retention_delta.png")
    plot_training_curves(series, out_dir / "training_losses.png")

    print(f"Saved analysis plots to {out_dir}")


if __name__ == "__main__":
    main()
