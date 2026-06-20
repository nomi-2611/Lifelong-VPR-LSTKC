import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        parsed.append({k: float(v) for k, v in row.items()})
    parsed.sort(key=lambda item: item["weight"])
    return parsed


def annotate_best(ax, xs, ys, label, color, paper=False):
    best_idx = max(range(len(ys)), key=lambda idx: ys[idx])
    x = xs[best_idx]
    y = ys[best_idx]
    ax.scatter([x], [y], color=color, s=48 if paper else 60, zorder=5)
    ax.annotate(
        f"{label} best @ {x:.2f}: {y:.1f}",
        xy=(x, y),
        xytext=(10, 10 if paper else 12),
        textcoords="offset points",
        fontsize=9 if paper else 10,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )


def build_style(paper=False):
    if paper:
        return {
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#c7d1db",
            "grid.color": "#d9e1ea",
            "grid.linestyle": "--",
            "grid.alpha": 0.75,
            "font.size": 10,
        }
    return {
        "axes.facecolor": "#fbfcfe",
        "figure.facecolor": "#f4f7fb",
        "axes.edgecolor": "#c9d3df",
        "grid.color": "#dfe6ef",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.size": 11,
    }


def plot(csv_path, out_path, paper=False):
    rows = load_rows(csv_path)
    xs = [row["weight"] for row in rows]

    plt.rcParams.update(build_style(paper=paper))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.8) if paper else (14, 5.8),
        dpi=320 if paper else 220,
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(xs, [row["seen_avg_mAP"] for row in rows], marker="o", linewidth=2.3, color="#1f77b4", label="Seen Avg mAP")
    ax.plot(xs, [row["seen_avg_R1"] for row in rows], marker="s", linewidth=2.3, color="#ff7f0e", label="Seen Avg R1")
    ax.plot(xs, [row["unseen_avg_mAP"] for row in rows], marker="^", linewidth=1.9, color="#2ca02c", label="Unseen Avg mAP")
    ax.plot(xs, [row["unseen_avg_R1"] for row in rows], marker="D", linewidth=1.9, color="#d62728", label="Unseen Avg R1")
    annotate_best(ax, xs, [row["seen_avg_mAP"] for row in rows], "mAP", "#1f77b4", paper=paper)
    annotate_best(ax, xs, [row["seen_avg_R1"] for row in rows], "R1", "#ff7f0e", paper=paper)
    ax.set_title("A. Average Metrics vs Fuse Weight", loc="left", fontsize=12 if paper else 13, fontweight="bold")
    ax.set_xlabel("Model Distance Weight")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.legend(frameon=True, ncol=2, fontsize=9 if paper else 10)

    ax = axes[1]
    ax.plot(xs, [row["market1501_mAP"] for row in rows], marker="o", linewidth=2.0, color="#4e79a7", label="market1501 mAP")
    ax.plot(xs, [row["cuhk_sysu_mAP"] for row in rows], marker="s", linewidth=2.0, color="#f28e2b", label="cuhk_sysu mAP")
    ax.plot(xs, [row["dukemtmc_mAP"] for row in rows], marker="^", linewidth=2.0, color="#59a14f", label="dukemtmc mAP")
    ax.scatter([xs[-1]], [rows[-1]["market1501_mAP"]], color="#4e79a7", s=36 if paper else 46, zorder=5)
    ax.set_title("B. Per-dataset mAP vs Fuse Weight", loc="left", fontsize=12 if paper else 13, fontweight="bold")
    ax.set_xlabel("Model Distance Weight")
    ax.set_ylabel("mAP (%)")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.legend(frameon=True, fontsize=9 if paper else 10)

    best = max(rows, key=lambda row: row["seen_avg_mAP"])
    title = (
        "VPRTempo Fuse Ablation"
        if paper
        else "VPRTempo Fuse Ablation\n"
    )
    subtitle = f"Best weight = {best['weight']:.2f} | Seen Avg {best['seen_avg_mAP']:.1f} mAP / {best['seen_avg_R1']:.1f} R1"
    fig.suptitle(title + subtitle if not paper else title, fontsize=14 if paper else 15, fontweight="bold", y=1.02 if paper else 0.98)

    if paper:
        fig.text(
            0.5,
            0.94,
            subtitle,
            ha="center",
            fontsize=10.5,
            color="#33485d",
        )

    note = (
        f"Best detailed result @ {best['weight']:.2f}: "
        f"market1501 {best['market1501_mAP']:.1f}/{best['market1501_R1']:.1f}, "
        f"cuhk_sysu {best['cuhk_sysu_mAP']:.1f}/{best['cuhk_sysu_R1']:.1f}, "
        f"dukemtmc {best['dukemtmc_mAP']:.1f}/{best['dukemtmc_R1']:.1f}."
    )
    fig.text(0.01, 0.01, note, fontsize=9 if paper else 10, color="#445266")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_split(csv_path, out_dir, paper=False):
    rows = load_rows(csv_path)
    xs = [row["weight"] for row in rows]
    best = max(rows, key=lambda row: row["seen_avg_mAP"])

    plt.rcParams.update(build_style(paper=paper))
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(6.6, 4.8) if paper else (7.4, 5.4),
        dpi=320 if paper else 220,
        constrained_layout=True,
    )
    ax.plot(xs, [row["seen_avg_mAP"] for row in rows], marker="o", linewidth=2.3, color="#1f77b4", label="Seen Avg mAP")
    ax.plot(xs, [row["seen_avg_R1"] for row in rows], marker="s", linewidth=2.3, color="#ff7f0e", label="Seen Avg R1")
    ax.plot(xs, [row["unseen_avg_mAP"] for row in rows], marker="^", linewidth=1.9, color="#2ca02c", label="Unseen Avg mAP")
    ax.plot(xs, [row["unseen_avg_R1"] for row in rows], marker="D", linewidth=1.9, color="#d62728", label="Unseen Avg R1")
    annotate_best(ax, xs, [row["seen_avg_mAP"] for row in rows], "mAP", "#1f77b4", paper=paper)
    annotate_best(ax, xs, [row["seen_avg_R1"] for row in rows], "R1", "#ff7f0e", paper=paper)
    ax.set_title("Average Metrics vs Fuse Weight", fontsize=12 if paper else 13, fontweight="bold")
    ax.set_xlabel("Model Distance Weight")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.legend(frameon=True, ncol=2, fontsize=9 if paper else 10)
    avg_path = out_dir / ("fuse_ablation_average_paper_hd.png" if paper else "fuse_ablation_average.png")
    fig.savefig(avg_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(6.6, 4.8) if paper else (7.4, 5.4),
        dpi=320 if paper else 220,
        constrained_layout=True,
    )
    ax.plot(xs, [row["market1501_mAP"] for row in rows], marker="o", linewidth=2.0, color="#4e79a7", label="market1501 mAP")
    ax.plot(xs, [row["cuhk_sysu_mAP"] for row in rows], marker="s", linewidth=2.0, color="#f28e2b", label="cuhk_sysu mAP")
    ax.plot(xs, [row["dukemtmc_mAP"] for row in rows], marker="^", linewidth=2.0, color="#59a14f", label="dukemtmc mAP")
    ax.scatter([best["weight"]], [best["market1501_mAP"]], color="#4e79a7", s=36 if paper else 46, zorder=5)
    ax.set_title("Per-dataset mAP vs Fuse Weight", fontsize=12 if paper else 13, fontweight="bold")
    ax.set_xlabel("Model Distance Weight")
    ax.set_ylabel("mAP (%)")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.legend(frameon=True, fontsize=9 if paper else 10)
    dataset_path = out_dir / ("fuse_ablation_per_dataset_paper_hd.png" if paper else "fuse_ablation_per_dataset.png")
    fig.savefig(dataset_path, bbox_inches="tight")
    plt.close(fig)

    return avg_path, dataset_path


def main():
    parser = argparse.ArgumentParser(description="Plot fuse ablation curves from a summary CSV.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split-dir", default=None, help="Optional directory to also save two split single-panel figures")
    parser.add_argument("--paper", action="store_true", help="Generate a tighter high-resolution paper-style figure")
    args = parser.parse_args()

    csv_path = Path(args.summary_csv)
    default_name = "fuse_ablation_curve_paper_hd.png" if args.paper else "fuse_ablation_curve.png"
    out_path = Path(args.output) if args.output else csv_path.with_name(default_name)
    plot(csv_path, out_path, paper=args.paper)
    print(f"Saved fuse ablation curve to {out_path}")
    if args.split_dir:
        avg_path, dataset_path = plot_split(csv_path, Path(args.split_dir), paper=args.paper)
        print(f"Saved split fuse ablation figures to {avg_path} and {dataset_path}")


if __name__ == "__main__":
    main()
