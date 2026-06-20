from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path
from types import SimpleNamespace

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

from vprtempo_tools.export_qg_matrix import (
    ExplicitPathImageDataset,
    ProcessImage,
    build_models,
    extract_embeddings,
    infer_model_structure,
    model_logger,
)


def load_path_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(similarity, gt, topk=(1, 5, 10)):
    order = np.argsort(-similarity, axis=1)
    valid = gt.any(axis=1)
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise RuntimeError("No valid queries in GT")

    ap_scores = []
    recalls = {int(k): 0.0 for k in topk}
    for query_index in range(similarity.shape[0]):
        if not valid[query_index]:
            continue
        ranked_matches = gt[query_index, order[query_index]].astype(bool)
        positive_ranks = np.flatnonzero(ranked_matches)
        precisions = [(rank_idx + 1.0) / float(rank + 1.0) for rank_idx, rank in enumerate(positive_ranks)]
        ap_scores.append(float(np.mean(precisions)))
        for k in recalls:
            recalls[k] += float(ranked_matches[:k].any())

    result = {
        "valid_queries": valid_count,
        "mAP": float(np.mean(ap_scores)),
    }
    for k in topk:
        result[f"R@{k}"] = recalls[int(k)] / float(valid_count)
    return result


def export_embeddings(model_path, dims, patches, workers, query_paths, gallery_paths):
    structure = infer_model_structure(model_path)
    runtime_args = SimpleNamespace(
        dataset="tart_minimal_protocol",
        data_dir=".",
        database_places=structure["database_places"],
        query_places=len(query_paths),
        max_module=max(structure["out_dims"]),
        database_dirs="protocol_gallery",
        query_dir="protocol_query",
        GT_tolerance=0,
        skip=0,
        filter=1,
        epoch=1,
        patches=patches,
        dims=",".join(str(x) for x in dims),
        train_new_model=False,
        quantize=False,
        PR_curve=False,
        sim_mat=False,
        run_demo=False,
        export_matrix=False,
        export_prefix="tart_protocol",
    )
    logger, output_folder = model_logger()
    models = build_models(runtime_args, dims, logger, output_folder, structure)
    models[0].load_model(models, model_path)
    image_transform = ProcessImage(dims, patches)

    query_embeddings = extract_embeddings(
        models,
        ExplicitPathImageDataset(query_paths, transform=image_transform),
        workers,
    ).cpu().numpy().astype(np.float32)
    gallery_embeddings = extract_embeddings(
        models,
        ExplicitPathImageDataset(gallery_paths, transform=image_transform),
        workers,
    ).cpu().numpy().astype(np.float32)

    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
    gallery_embeddings /= np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-12
    return query_embeddings, gallery_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TART minimal protocol with raw VPRTempo embeddings.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dims", default="56,56")
    parser.add_argument("--patches", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    protocol_dir = Path(args.protocol_dir).resolve()
    scene = args.scene

    query_paths = load_path_list(protocol_dir / f"{scene}_query_paths.json")
    gallery_paths = load_path_list(protocol_dir / f"{scene}_gallery_paths.json")
    gt = np.asarray(np.load(protocol_dir / f"{scene}_gt.npy"), dtype=np.uint8)

    query_embeddings, gallery_embeddings = export_embeddings(
        model_path=args.model_path,
        dims=dims,
        patches=args.patches,
        workers=args.workers,
        query_paths=query_paths,
        gallery_paths=gallery_paths,
    )
    similarity = np.matmul(query_embeddings, gallery_embeddings.T)
    metrics = compute_metrics(similarity, gt, topk=(1, 5, 10))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{scene}_query_embeddings.npy", query_embeddings)
    np.save(output_dir / f"{scene}_gallery_embeddings.npy", gallery_embeddings)
    np.save(output_dir / f"{scene}_similarity.npy", similarity.astype(np.float32))

    result = {
        "scene": scene,
        "query_count": int(query_embeddings.shape[0]),
        "gallery_count": int(gallery_embeddings.shape[0]),
        **metrics,
    }
    result_path = output_dir / f"{scene}_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


