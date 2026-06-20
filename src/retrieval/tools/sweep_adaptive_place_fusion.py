import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_weights(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_pair_list(raw):
    pairs = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        left, right = token.split(":")
        pairs.append((float(left), float(right)))
    return pairs


def extract_metrics(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result_line = None
    for line in reversed(lines):
        if line.startswith("|") and "/" in line:
            result_line = line
            break
    if result_line is None:
        raise RuntimeError(f"Could not find metrics line in {log_path}")
    values = [chunk for chunk in result_line.split("|") if chunk]
    labels = ["nordland_place", "robotcar_place", "tart_place", "average"]
    metrics = {}
    for label, value in zip(labels, values):
        m_ap, r1 = value.strip().split("/")
        metrics[f"{label}_mAP"] = float(m_ap)
        metrics[f"{label}_R1"] = float(r1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Sweep confidence-adaptive place fusion settings.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-folder", required=True)
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--temps", default="0.5,1,2,5")
    parser.add_argument("--bounds", default="0.5:0.98,0.6:0.98,0.7:0.98")
    parser.add_argument("--modes", default="gap,gap_entropy,gap_entropy_max")
    parser.add_argument("--eval-query-limit", type=int, default=128)
    parser.add_argument("--eval-gallery-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    logs_root = Path(args.logs_root).resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    temps = parse_float_list(args.temps)
    bounds = parse_pair_list(args.bounds)
    modes = parse_weights(args.modes)

    rows = []
    for temp, (alpha_min, alpha_max), mode in itertools.product(temps, bounds, modes):
        run_name = f"adaptive_{mode}_t{str(temp).replace('.', 'p')}_a{str(alpha_min).replace('.', 'p')}-{str(alpha_max).replace('.', 'p')}"
        logs_dir = logs_root / run_name
        cmd = [
            sys.executable, "continual_train.py",
            "--task-type", "place",
            "--data-dir", str(Path(args.data_dir).resolve()),
            "--logs-dir", str(logs_dir),
            "--test_folder", str(Path(args.test_folder).resolve()),
            "--evaluate",
            "--workers", str(args.workers),
            "--batch-size", str(args.batch_size),
            "--vprtempo-embedding-manifest", str(Path(args.embedding_manifest).resolve()),
            "--vprtempo-embedding-mode", "fuse",
            "--vprtempo-embedding-weight", "0.5",
            "--vprtempo-fusion-strategy", "confidence_adaptive",
            "--vprtempo-adaptive-alpha-temp", str(temp),
            "--vprtempo-adaptive-confidence-mode", mode,
            "--vprtempo-adaptive-alpha-min", str(alpha_min),
            "--vprtempo-adaptive-alpha-max", str(alpha_max),
            "--eval-query-limit", str(args.eval_query_limit),
            "--eval-gallery-limit", str(args.eval_gallery_limit),
        ]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        metrics = extract_metrics(logs_dir / "log_res.txt")
        row = {
            "run_name": run_name,
            "temp": temp,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "mode": mode,
        }
        row.update(metrics)
        rows.append(row)

    if args.dry_run:
        return

    summary_json = logs_root / "adaptive_sweep_summary.json"
    summary_csv = logs_root / "adaptive_sweep_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(summary_json)
    print(summary_csv)


if __name__ == "__main__":
    main()
