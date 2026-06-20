import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]
LSTKC_ROOT = ROOT.parent / "AAAI2024-LSTKC"
if str(LSTKC_ROOT) not in sys.path:
    sys.path.insert(0, str(LSTKC_ROOT))

import src.datasets.lreid_dataset.datasets as lstkc_datasets


def infer_relative_dir(data_root, samples):
    parents = {str(Path(sample[0]).resolve().parent) for sample in samples}
    if len(parents) != 1:
        raise ValueError(f"Expected a single image directory for this split, found {len(parents)}")
    image_dir = Path(next(iter(parents))).resolve()
    return str(image_dir.relative_to(Path(data_root).resolve())).replace("\\", "/")


def write_csv(csv_path, samples, label_source="index"):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("Image_name,index\n")
        for idx, sample in enumerate(samples):
            if label_source == "pid":
                label = int(sample[1])
            else:
                label = idx
            f.write(f"{Path(sample[0]).name},{label}\n")


def group_samples_by_parent(data_root, samples):
    groups = defaultdict(list)
    root = Path(data_root).resolve()
    for sample in samples:
        parent_rel = str(Path(sample[0]).resolve().parent.relative_to(root)).replace("\\", "/")
        groups[parent_rel].append(sample)
    return dict(sorted(groups.items()))


def main():
    parser = argparse.ArgumentParser(description="Create VPRTempo CSVs from exact LSTKC dataset splits.")
    parser.add_argument("--dataset-name", required=True, help="LSTKC dataset key, e.g. market1501 or dukemtmc")
    parser.add_argument("--data-root", default=str((LSTKC_ROOT / "PRID").resolve()), help="Root passed to the LSTKC dataset loader")
    parser.add_argument("--csv-dataset", required=True, help="Dataset prefix used in the generated VPRTempo CSV filename")
    parser.add_argument("--split", default="train", choices=["train", "query", "gallery"], help="Which LSTKC split to export")
    parser.add_argument("--csv-key", required=True, help="CSV key used in the generated filename and later passed as database_dirs")
    parser.add_argument("--output-dir", default=str((ROOT / "vprtempo" / "dataset").resolve()), help="Where to write the generated CSV")
    parser.add_argument("--label-source", default="index", choices=["index", "pid"], help="Write sequential row indices or preserve LSTKC pid labels")
    parser.add_argument("--group-by-parent", action="store_true", help="Write one CSV per parent image directory instead of assuming a single directory")
    args = parser.parse_args()

    dataset = lstkc_datasets.create(args.dataset_name, args.data_root)
    samples = getattr(dataset, args.split)
    output_dir = Path(args.output_dir)

    if args.group_by_parent:
        groups = group_samples_by_parent(args.data_root, samples)
        exports = []
        for parent_rel, group_samples in groups.items():
            dir_name = parent_rel.replace("\\", "_").replace("/", "_")
            csv_key = f"{args.csv_key}_{dir_name}"
            csv_path = output_dir / f"{args.csv_dataset}-{csv_key}.csv"
            write_csv(csv_path, group_samples, label_source=args.label_source)
            exports.append({
                "csv_key": csv_key,
                "csv_path": str(csv_path.resolve()).replace("\\", "/"),
                "image_dir": parent_rel,
                "num_samples": len(group_samples),
            })
        summary = {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "csv_dataset": args.csv_dataset,
            "base_csv_key": args.csv_key,
            "label_source": args.label_source,
            "group_by_parent": True,
            "num_samples": len(samples),
            "num_groups": len(exports),
            "exports": exports,
        }
        summary_path = output_dir / f"{args.csv_dataset}-{args.csv_key}.grouped.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    csv_path = output_dir / f"{args.csv_dataset}-{args.csv_key}.csv"
    write_csv(csv_path, samples, label_source=args.label_source)

    image_dir_rel = infer_relative_dir(args.data_root, samples)
    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "csv_path": str(csv_path.resolve()).replace("\\", "/"),
        "csv_dataset": args.csv_dataset,
        "csv_key": args.csv_key,
        "image_dir": image_dir_rel,
        "num_samples": len(samples),
        "label_source": args.label_source,
        "group_by_parent": False,
    }
    summary_path = csv_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
