import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log_res(path):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    candidate_indices = []
    for idx in range(len(lines) - 4):
        if (
            "|Average" in lines[idx]
            and "/" in lines[idx + 1]
            and "|Average" in lines[idx + 3]
            and "/" in lines[idx + 4]
        ):
            candidate_indices.append(idx)
    if not candidate_indices:
        raise ValueError(f"Unexpected log_res format: {path}")
    start = candidate_indices[-1]
    if start + 4 >= len(lines):
        raise ValueError(f"Incomplete result block in {path}")

    def clean_names(raw_line):
        names = []
        for token in raw_line.split("\t"):
            token = token.strip().strip("|")
            if not token or token == "Average":
                continue
            names.append(token)
        return names

    seen_names = clean_names(lines[start])
    seen_vals = [float(x) for x in re.findall(r"\d+\.\d+", lines[start + 1])]
    unseen_names = clean_names(lines[start + 3])
    unseen_vals = [float(x) for x in re.findall(r"\d+\.\d+", lines[start + 4])]

    seen = {}
    for idx, name in enumerate(seen_names[: max(0, (len(seen_vals) - 2) // 2)]):
        seen[name] = {"mAP": seen_vals[idx * 2], "R1": seen_vals[idx * 2 + 1]}
    seen_avg = {"mAP": seen_vals[-2], "R1": seen_vals[-1]}

    unseen = {}
    for idx, name in enumerate(unseen_names[: max(0, (len(unseen_vals) - 2) // 2)]):
        unseen[name] = {"mAP": unseen_vals[idx * 2], "R1": unseen_vals[idx * 2 + 1]}
    unseen_avg = {"mAP": unseen_vals[-2], "R1": unseen_vals[-1]}

    return {
        "seen": seen,
        "seen_avg": seen_avg,
        "unseen": unseen,
        "unseen_avg": unseen_avg,
    }


def load_results(baseline_run, matrix_run, marketadapt_run=None):
    baseline = parse_log_res(Path(baseline_run) / "log_res.txt")
    matrix = parse_log_res(Path(matrix_run) / "log_res.txt")
    marketadapt = parse_log_res(Path(marketadapt_run) / "log_res.txt") if marketadapt_run else None
    return baseline, matrix, marketadapt


def ensure_analysis_dir(run_dir):
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir


def annotate_bars(ax, bars, fmt="{:.1f}", fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(0.6, abs(h) * 0.02),
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#243447",
        )


def create_dashboard(baseline, matrix, marketadapt, output_path):
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.facecolor": "#fbfcfe",
            "figure.facecolor": "#f4f7fb",
            "axes.edgecolor": "#c9d3df",
            "grid.color": "#dfe6ef",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "font.size": 11,
        }
    )
    fig = plt.figure(figsize=(18, 12), dpi=200)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

    datasets = list(baseline["seen"].keys())
    x = np.arange(len(datasets))
    width = 0.32

    baseline_map = [baseline["seen"][d]["mAP"] for d in datasets]
    matrix_map = [matrix["seen"][d]["mAP"] for d in datasets]
    baseline_r1 = [baseline["seen"][d]["R1"] for d in datasets]
    matrix_r1 = [matrix["seen"][d]["R1"] for d in datasets]

    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x - width / 2, baseline_map, width, label="LSTKC Baseline", color="#5B8FF9")
    bars2 = ax1.bar(x + width / 2, matrix_map, width, label="VPRTempo Matrix Input", color="#E8684A")
    ax1.set_title("A. Dataset mAP Comparison", loc="left", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel("mAP (%)")
    ax1.legend(frameon=True)
    annotate_bars(ax1, bars1)
    annotate_bars(ax1, bars2)

    ax2 = fig.add_subplot(gs[0, 1])
    bars3 = ax2.bar(x - width / 2, baseline_r1, width, label="LSTKC Baseline", color="#5B8FF9")
    bars4 = ax2.bar(x + width / 2, matrix_r1, width, label="VPRTempo Matrix Input", color="#E8684A")
    ax2.set_title("B. Dataset Rank-1 Comparison", loc="left", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylabel("Rank-1 (%)")
    ax2.legend(frameon=True)
    annotate_bars(ax2, bars3)
    annotate_bars(ax2, bars4)

    ax3 = fig.add_subplot(gs[1, 0])
    setups = ["Baseline", "VPRTempo"]
    seen_avg_map = [baseline["seen_avg"]["mAP"], matrix["seen_avg"]["mAP"]]
    seen_avg_r1 = [baseline["seen_avg"]["R1"], matrix["seen_avg"]["R1"]]
    unseen_avg_map = [baseline["unseen_avg"]["mAP"], matrix["unseen_avg"]["mAP"]]
    unseen_avg_r1 = [baseline["unseen_avg"]["R1"], matrix["unseen_avg"]["R1"]]
    if marketadapt:
        setups.append("Market Warm-start")
        seen_avg_map.append(marketadapt["seen_avg"]["mAP"])
        seen_avg_r1.append(marketadapt["seen_avg"]["R1"])
        unseen_avg_map.append(marketadapt["unseen_avg"]["mAP"])
        unseen_avg_r1.append(marketadapt["unseen_avg"]["R1"])

    sx = np.arange(len(setups))
    w = 0.18
    b1 = ax3.bar(sx - 1.5 * w, seen_avg_map, w, label="Seen mAP", color="#5AD8A6")
    b2 = ax3.bar(sx - 0.5 * w, seen_avg_r1, w, label="Seen R1", color="#5B8FF9")
    b3 = ax3.bar(sx + 0.5 * w, unseen_avg_map, w, label="Unseen mAP", color="#F6BD16")
    b4 = ax3.bar(sx + 1.5 * w, unseen_avg_r1, w, label="Unseen R1", color="#E8684A")
    ax3.set_title("C. Average Seen / Unseen Performance", loc="left", fontsize=14, fontweight="bold")
    ax3.set_xticks(sx)
    ax3.set_xticklabels(setups)
    ax3.set_ylabel("Score (%)")
    ax3.legend(ncol=2, frameon=True)
    for bars in [b1, b2, b3, b4]:
        annotate_bars(ax3, bars, fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    delta_map = [matrix["seen"][d]["mAP"] - baseline["seen"][d]["mAP"] for d in datasets]
    delta_r1 = [matrix["seen"][d]["R1"] - baseline["seen"][d]["R1"] for d in datasets]
    d1 = ax4.bar(x - width / 2, delta_map, width, label="mAP delta", color="#C6E5FF")
    d2 = ax4.bar(x + width / 2, delta_r1, width, label="R1 delta", color="#FFC9C0")
    ax4.axhline(0, color="#5c6773", linewidth=1.2)
    ax4.set_title("D. Relative Change vs Baseline", loc="left", fontsize=14, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.set_ylabel("Delta (%)")
    ax4.legend(frameon=True)
    annotate_bars(ax4, d1)
    annotate_bars(ax4, d2)

    best_matrix = max(matrix["seen"].items(), key=lambda item: item[1]["mAP"])
    worst_matrix = min(matrix["seen"].items(), key=lambda item: item[1]["mAP"])
    headline = (
        "VPRTempo Matrix Input Overview\n"
        f"All-dataset matrix replacement run: Seen average {matrix['seen_avg']['mAP']:.1f} mAP / {matrix['seen_avg']['R1']:.1f} R1, "
        f"vs baseline {baseline['seen_avg']['mAP']:.1f} / {baseline['seen_avg']['R1']:.1f}. "
        f"Best matrix-input dataset: {best_matrix[0]} ({best_matrix[1]['mAP']:.1f}/{best_matrix[1]['R1']:.1f}); "
        f"weakest: {worst_matrix[0]} ({worst_matrix[1]['mAP']:.1f}/{worst_matrix[1]['R1']:.1f})."
    )
    fig.suptitle(headline, fontsize=16, fontweight="bold", y=0.98)

    note_lines = [
        "Runs: baseline=logs/run_full_3stage, matrix=logs/run_vprtempo_all3_formal_aligned",
        "Matrix source: VPRTempo exported qg similarity matrices aligned to LSTKC query/gallery ordering.",
    ]
    if marketadapt:
        note_lines.append(
            f"Additional ablation: Market warm-start produced {marketadapt['seen']['market1501']['mAP']:.1f} mAP / "
            f"{marketadapt['seen']['market1501']['R1']:.1f} R1 on market1501."
        )
    fig.text(0.01, 0.015, "\n".join(note_lines), fontsize=10, color="#445266")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_summary(baseline, matrix, marketadapt, output_path):
    summary = {
        "baseline_seen_avg": baseline["seen_avg"],
        "matrix_seen_avg": matrix["seen_avg"],
        "baseline_unseen_avg": baseline["unseen_avg"],
        "matrix_unseen_avg": matrix["unseen_avg"],
        "per_dataset_baseline": baseline["seen"],
        "per_dataset_matrix": matrix["seen"],
    }
    if marketadapt:
        summary["market_warmstart_seen_avg"] = marketadapt["seen_avg"]
        summary["market_warmstart_market1501"] = marketadapt["seen"]["market1501"]
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create a dashboard for VPRTempo matrix-input LSTKC runs.")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--matrix-run", required=True)
    parser.add_argument("--marketadapt-run", default=None)
    args = parser.parse_args()

    baseline, matrix, marketadapt = load_results(args.baseline_run, args.matrix_run, args.marketadapt_run)
    analysis_dir = ensure_analysis_dir(args.matrix_run)
    dashboard_path = analysis_dir / "vprtempo_matrix_dashboard_overview.png"
    summary_path = analysis_dir / "vprtempo_matrix_dashboard_summary.json"

    create_dashboard(baseline, matrix, marketadapt, dashboard_path)
    save_summary(baseline, matrix, marketadapt, summary_path)

    print(f"Saved dashboard to {dashboard_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
