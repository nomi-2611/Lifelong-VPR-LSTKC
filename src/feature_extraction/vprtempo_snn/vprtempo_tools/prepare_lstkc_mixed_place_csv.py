import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LSTKC_ROOT = ROOT.parent / "AAAI2024-LSTKC"
if str(LSTKC_ROOT) not in sys.path:
    sys.path.insert(0, str(LSTKC_ROOT))

import src.datasets.lreid_dataset.datasets as lstkc_datasets


def sample_split(dataset_name, data_root, split_name, per_dataset_limit=None, stride=1):
    dataset = lstkc_datasets.create(dataset_name, data_root)
    samples = list(getattr(dataset, split_name))
    samples = samples[::max(1, stride)]
    if per_dataset_limit is not None:
        samples = samples[:per_dataset_limit]
    return samples


def interleave_groups(grouped_samples):
    mixed = []
    max_len = max((len(samples) for _, samples in grouped_samples), default=0)
    for idx in range(max_len):
        for dataset_name, samples in grouped_samples:
            if idx < len(samples):
                mixed.append((dataset_name, samples[idx]))
    return mixed


def write_csv(csv_path, data_root, mixed_samples):
    root = Path(data_root).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("Image_name,index\n")
        for global_idx, (_, sample) in enumerate(mixed_samples):
            rel_path = str(Path(sample[0]).resolve().relative_to(root)).replace("\\", "/")
            f.write(f"{rel_path},{global_idx}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced mixed VPRTempo CSV from multiple LSTKC place datasets."
    )
    parser.add_argument("--dataset-names", nargs="+", default=["nordland_place", "robotcar_place", "tart_place"],
                        help="LSTKC dataset keys to mix")
    parser.add_argument("--dataset-weights", nargs="+", type=int, default=None,
                        help="Optional integer sampling weights aligned with --dataset-names, e.g. 1 1 2")
    parser.add_argument("--data-root", required=True, help="Root passed to the LSTKC dataset loader")
    parser.add_argument("--split", default="train", choices=["train", "query", "gallery"],
                        help="Which LSTKC split to export")
    parser.add_argument("--per-dataset-limit", type=int, default=1024,
                        help="Maximum number of samples drawn from each dataset before mixing")
    parser.add_argument("--stride", type=int, default=1,
                        help="Optional sub-sampling stride applied within each dataset before limiting")
    parser.add_argument("--csv-dataset", default="place_mix",
                        help="Dataset prefix used in the generated VPRTempo CSV filename")
    parser.add_argument("--csv-key", default="balanced",
                        help="CSV key used in the generated filename and passed later as database_dirs")
    parser.add_argument("--output-dir", default=str((ROOT / "vprtempo" / "dataset").resolve()),
                        help="Where to write the generated CSV")
    args = parser.parse_args()

    if args.dataset_weights is None:
        dataset_weights = [1] * len(args.dataset_names)
    else:
        if len(args.dataset_weights) != len(args.dataset_names):
            raise ValueError("--dataset-weights must have the same length as --dataset-names")
        dataset_weights = [max(1, int(x)) for x in args.dataset_weights]

    grouped = []
    counts = {}
    weighted_counts = {}
    for dataset_name, weight in zip(args.dataset_names, dataset_weights):
        samples = sample_split(
            dataset_name=dataset_name,
            data_root=args.data_root,
            split_name=args.split,
            per_dataset_limit=args.per_dataset_limit * weight,
            stride=args.stride,
        )
        grouped.append((dataset_name, samples))
        counts[dataset_name] = len(samples)
        weighted_counts[dataset_name] = weight

    mixed_samples = interleave_groups(grouped)
    csv_path = Path(args.output_dir) / f"{args.csv_dataset}-{args.csv_key}.csv"
    write_csv(csv_path, args.data_root, mixed_samples)

    summary = {
        "dataset_names": args.dataset_names,
        "dataset_weights": weighted_counts,
        "split": args.split,
        "per_dataset_limit": args.per_dataset_limit,
        "stride": args.stride,
        "csv_path": str(csv_path.resolve()).replace("\\", "/"),
        "csv_dataset": args.csv_dataset,
        "csv_key": args.csv_key,
        "image_dir": ".",
        "num_samples": len(mixed_samples),
        "counts": counts,
    }
    summary_path = csv_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
