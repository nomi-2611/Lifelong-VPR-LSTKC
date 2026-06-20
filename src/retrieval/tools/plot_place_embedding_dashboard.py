from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log_res(log_res_path: Path):
    lines = [line.strip() for line in log_res_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Unexpected log_res format: {log_res_path}")
    dataset_names = [item.strip() for item in lines[0].split("\t") if item.strip() and item.strip() != "|Average\t|" and item.strip() != "|Average|"]
    raw_metrics = [item for item in lines[1].split("|") if item.strip()]
    dataset_metrics = raw_metrics[:-1]
    average_metric = raw_metrics[-1]
    results = {}
    for name, metric in zip(dataset_names, dataset_metrics):
        mAP_str, r1_str = metric.strip().split("/")
        results[name] = {"mAP": float(mAP_str), "R1": float(r1_str)}
    avg_map, avg_r1 = average_metric.strip().split("/")
    return results, {"mAP": float(avg_map), "R1": float(avg_r1)}


def load_embedding_summaries(paths):
    summary = {}
    for path in paths:
        item = json.loads(Path(path).read_text(encoding="utf-8"))
        summary[item["query_dir"]] = item
    return summary


def build_dashboard(run_log_res, embedding_summaries, output_png, output_json):
    metrics, average = parse_log_res(Path(run_log_res))
    emb = load_embedding_summaries(embedding_summaries)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    datasets = list(metrics.keys())
    mAP = [metrics[name]["mAP"] for name in datasets]
    r1 = [metrics[name]["R1"] for name in datasets]
    q_counts = [emb[name]["query_places"] for name in datasets]
    g_counts = [emb[name]["gallery_places"] for name in datasets]

    summary = {
        "datasets": [
            {
                "name": name,
                "mAP": metrics[name]["mAP"],
                "R1": metrics[name]["R1"],
                "query_places": emb[name]["query_places"],
                "gallery_places": emb[name]["gallery_places"],
                "embedding_dim": emb[name]["query_embedding_shape"][1],
            }
            for name in datasets
        ],
        "average": average,
        "notes": {
            "evaluation_mode": "continual_train.py --task-type place --evaluate",
            "feature_source": "VPRTempo external query/gallery embeddings",
            "metric": "cosine distance over normalized embeddings",
        },
    }
    Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    fig = plt.figure(figsize=(16, 10), dpi=220)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(datasets))
    width = 0.36
    bars1 = ax1.bar(x - width / 2, mAP, width=width, label="mAP", color="#1f77b4")
    bars2 = ax1.bar(x + width / 2, r1, width=width, label="R@1", color="#ff7f0e")
    ax1.set_title("Place Embedding Performance")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=12)
    ax1.legend(frameon=True)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    counts_x = np.arange(len(datasets))
    ax2.bar(counts_x - width / 2, q_counts, width=width, label="Queries", color="#2ca02c")
    ax2.bar(counts_x + width / 2, g_counts, width=width, label="Gallery", color="#9467bd")
    ax2.set_title("Evaluation Sample Counts")
    ax2.set_ylabel("Images")
    ax2.set_xticks(counts_x)
    ax2.set_xticklabels(datasets, rotation=12)
    ax2.legend(frameon=True)

    ax3 = fig.add_subplot(gs[1, 0])
    avg_names = ["Average mAP", "Average R@1"]
    avg_values = [average["mAP"], average["R1"]]
    avg_colors = ["#1f77b4", "#ff7f0e"]
    bars = ax3.bar(avg_names, avg_values, color=avg_colors, width=0.55)
    ax3.set_title("Overall Average")
    ax3.set_ylabel("Score (%)")
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", va="bottom", fontsize=10)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    best_dataset = datasets[int(np.argmax(mAP))]
    key_lines = [
        "Place Recognition via External VPRTempo Embeddings",
        "",
        f"Best dataset: {best_dataset} ({metrics[best_dataset]['mAP']:.1f} mAP / {metrics[best_dataset]['R1']:.1f} R@1)",
        f"Average: {average['mAP']:.1f} mAP / {average['R1']:.1f} R@1",
        "Evaluation path: continual_train.py --task-type place --evaluate",
        "External signal: normalized VPRTempo query/gallery embeddings",
        "Sample cap: 512 queries / 1024 gallery per dataset",
        "Current bottleneck: tart_place transfers poorly under Nordland-pretrained weights",
    ]
    ax4.text(
        0.03,
        0.97,
        "\n".join(key_lines),
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f6f6f6", edgecolor="#cfcfcf"),
    )

    fig.suptitle("LSTKC Place Evaluation with VPRTempo Embedding Input", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "Embedding source: VPRTempo | Metrics from LSTKC evaluator in place mode", ha="center", fontsize=11)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build a dashboard for place evaluation with external VPRTempo embeddings.")
    parser.add_argument("--run-log-res", required=True)
    parser.add_argument("--embedding-summaries", nargs="+", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    build_dashboard(args.run_log_res, args.embedding_summaries, args.output_png, args.output_json)


if __name__ == "__main__":
    main()
