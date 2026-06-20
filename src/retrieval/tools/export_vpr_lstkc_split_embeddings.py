from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from collections import OrderedDict

import numpy as np


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
CODE_ROOT = osp.dirname(PROJECT_ROOT)
VPRTEMPO_ROOT = osp.join(CODE_ROOT, "feature_extraction", "vprtempo_snn")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if VPRTEMPO_ROOT not in sys.path:
    sys.path.insert(0, VPRTEMPO_ROOT)

import src.datasets.lreid_dataset.datasets as lstkc_datasets
from vprtempo_tools.export_qg_matrix import (
    ExplicitPathImageDataset,
    ProcessImage,
    build_models,
    extract_embeddings,
    infer_model_structure,
    model_logger,
)
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description="Export VPRTempo embeddings for arbitrary LSTKC splits.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-names", nargs="+", default=["nordland_place", "robotcar_place", "tart_place"])
    parser.add_argument("--split", default="train", choices=["train", "query", "gallery"])
    parser.add_argument("--database-dirs", default="balanced1024")
    parser.add_argument("--dims", default="56,56")
    parser.add_argument("--patches", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--per-dataset-limit", type=int, default=4096)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    structure = infer_model_structure(args.model_path)
    runtime_args = SimpleNamespace(
        dataset="place_formal",
        data_dir=args.data_root,
        database_places=structure["database_places"],
        query_places=0,
        max_module=max(structure["out_dims"]),
        database_dirs=args.database_dirs,
        query_dir="query",
        GT_tolerance=0,
        skip=0,
        filter=1,
        epoch=1,
        patches=args.patches,
        dims=",".join(str(x) for x in dims),
        train_new_model=False,
        quantize=False,
        PR_curve=False,
        sim_mat=False,
        run_demo=False,
        export_matrix=False,
        export_prefix="formal_split",
    )

    logger, output_folder = model_logger()
    models = build_models(runtime_args, dims, logger, output_folder, structure)
    models[0].load_model(models, args.model_path)
    image_transform = ProcessImage(dims, args.patches)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = OrderedDict()
    summary = OrderedDict()
    for dataset_name in args.dataset_names:
        dataset = lstkc_datasets.create(dataset_name, args.data_root)
        split = list(getattr(dataset, args.split))
        if args.per_dataset_limit is not None and args.per_dataset_limit > 0:
            split = split[:args.per_dataset_limit]
        image_paths = [osp.abspath(sample[0]) for sample in split]
        if not image_paths:
            raise ValueError(f"{dataset_name} {args.split} split is empty")

        embeddings = extract_embeddings(
            models,
            ExplicitPathImageDataset(image_paths, transform=image_transform),
            args.workers,
        )
        embeddings = embeddings.cpu().numpy().astype(np.float32)

        emb_path = osp.join(args.output_dir, f"{dataset_name}_{args.split}_embeddings.npy")
        np.save(emb_path, embeddings)
        manifest[dataset_name] = {
            f"{args.split}_embeddings": osp.abspath(emb_path).replace("\\", "/"),
            "metric": "cosine",
            "normalized": True,
        }
        summary[dataset_name] = {
            "split": args.split,
            "num_embeddings": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]),
            "path": osp.abspath(emb_path).replace("\\", "/"),
        }

    manifest_path = osp.join(args.output_dir, f"{args.split}_embedding_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary_path = osp.join(args.output_dir, f"{args.split}_embedding_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "manifest_path": osp.abspath(manifest_path).replace("\\", "/"),
        "summary_path": osp.abspath(summary_path).replace("\\", "/"),
        "per_dataset_limit": args.per_dataset_limit,
        "split": args.split,
    }, indent=2))


if __name__ == "__main__":
    main()


