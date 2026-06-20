from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = ["nordland_place", "robotcar_place", "tart_place"]


def parse_log_res(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics_line = lines[1]
    items = [item for item in metrics_line.split("|") if item.strip()]
    dataset_items = items[:-1]
    avg_item = items[-1]
    parsed = {}
    for name, item in zip(DATASETS, dataset_items):
        m, r = item.strip().split("/")
        parsed[name] = {"mAP": float(m), "R1": float(r)}
    avg_m, avg_r = avg_item.strip().split("/")
    parsed["average"] = {"mAP": float(avg_m), "R1": float(avg_r)}
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot place embedding/matrix fuse ablation.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--tag-prefix", default="w")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = []
    series = {name: [] for name in DATASETS + ["average"]}
    weight_values = []

    for weight in args.weights:
        tag = args.tag_prefix + weight.replace(".", "p")
        log_res = Path(args.run_root) / tag / "log_res.txt"
        parsed = parse_log_res(log_res)
        w = float(weight)
        weight_values.append(w)
        row = {"weight": w}
        for name in DATASETS + ["average"]:
            row[f"{name}_mAP"] = parsed[name]["mAP"]
            row[f"{name}_R1"] = parsed[name]["R1"]
            series[name].append(parsed[name])
        rows.append(row)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=220)

    for name in DATASETS + ["average"]:
        axes[0].plot(weight_values, [item["mAP"] for item in series[name]], marker="o", label=name)
        axes[1].plot(weight_values, [item["R1"] for item in series[name]], marker="o", label=name)

    axes[0].set_title("mAP vs Fusion Weight")
    axes[0].set_xlabel("Embedding Weight")
    axes[0].set_ylabel("mAP (%)")
    axes[1].set_title("R@1 vs Fusion Weight")
    axes[1].set_xlabel("Embedding Weight")
    axes[1].set_ylabel("R@1 (%)")
    axes[0].legend(frameon=True, fontsize=8)
    axes[1].legend(frameon=True, fontsize=8)
    fig.suptitle("Place Embedding/Matrix Fusion Ablation", fontsize=16, fontweight="bold")
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
