import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LSTKC_ROOT = ROOT.parent / "AAAI2024-LSTKC"
if str(LSTKC_ROOT) not in sys.path:
    sys.path.insert(0, str(LSTKC_ROOT))

import src.datasets.lreid_dataset.datasets as lstkc_datasets


def write_global_csv(csv_path, data_root, samples, label_source="pid"):
    root = Path(data_root).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("Image_name,index\n")
        for idx, sample in enumerate(samples):
            rel_path = str(Path(sample[0]).resolve().relative_to(root)).replace("\\", "/")
            label = int(sample[1]) if label_source == "pid" else idx
            f.write(f"{rel_path},{label}\n")


def main():
    parser = argparse.ArgumentParser(description="Create a single VPRTempo CSV from an LSTKC split using full relative image paths.")
    parser.add_argument("--dataset-name", required=True, help="LSTKC dataset key, e.g. robotcar_place or tart_place")
    parser.add_argument("--data-root", required=True, help="Root passed to the LSTKC dataset loader")
    parser.add_argument("--csv-dataset", required=True, help="Dataset prefix used in the generated VPRTempo CSV filename")
    parser.add_argument("--split", default="train", choices=["train", "query", "gallery"], help="Which LSTKC split to export")
    parser.add_argument("--csv-key", required=True, help="CSV key used in the generated filename and passed later as database_dirs")
    parser.add_argument("--output-dir", default=str((ROOT / "vprtempo" / "dataset").resolve()), help="Where to write the generated CSV")
    parser.add_argument("--label-source", default="pid", choices=["pid", "index"], help="Preserve LSTKC pid labels or rewrite as contiguous row indices")
    args = parser.parse_args()

    dataset = lstkc_datasets.create(args.dataset_name, args.data_root)
    samples = getattr(dataset, args.split)
    csv_path = Path(args.output_dir) / f"{args.csv_dataset}-{args.csv_key}.csv"
    write_global_csv(csv_path, args.data_root, samples, label_source=args.label_source)

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "csv_path": str(csv_path.resolve()).replace("\\", "/"),
        "csv_dataset": args.csv_dataset,
        "csv_key": args.csv_key,
        "label_source": args.label_source,
        "image_dir": ".",
        "num_samples": len(samples),
    }
    summary_path = csv_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
