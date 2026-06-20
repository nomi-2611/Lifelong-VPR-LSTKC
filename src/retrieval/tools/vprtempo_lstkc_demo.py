import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluators import Evaluator, evaluate_all


def normalize_similarity_to_distance(similarity):
    similarity = similarity.astype(np.float32)
    min_v = float(similarity.min())
    max_v = float(similarity.max())
    if max_v - min_v < 1e-12:
        return np.ones_like(similarity, dtype=np.float32)
    similarity = (similarity - min_v) / (max_v - min_v)
    return 1.0 - similarity


def build_query_gallery_from_gt(gt):
    gt = np.asarray(gt)
    query = []
    gallery = []
    for g_idx in range(gt.shape[0]):
        gallery.append((f"gallery_{g_idx:04d}.png", g_idx, 1, 0))
    for q_idx in range(gt.shape[1]):
        matched_gallery = int(np.argmax(gt[:, q_idx]))
        query.append((f"query_{q_idx:04d}.png", matched_gallery, 0, 0))
    return query, gallery


def main():
    parser = argparse.ArgumentParser(description="Demo bridge: evaluate a VPRTempo similarity matrix with LSTKC metrics.")
    parser.add_argument("--similarity", required=True, help="Path to VPRTempo similarity matrix .npy")
    parser.add_argument("--gt", required=True, help="Path to VPRTempo GT matrix .npy")
    parser.add_argument("--manifest", default=None, help="Optional LSTKC manifest JSON to exercise the formal external-matrix interface")
    parser.add_argument("--dataset-name", default="vprtempo_demo", help="Dataset name used inside the manifest")
    parser.add_argument("--out", default=None, help="Optional result text file path")
    args = parser.parse_args()

    similarity = np.load(args.similarity)
    gt = np.load(args.gt)
    query, gallery = build_query_gallery_from_gt(gt)
    if args.manifest:
        evaluator = Evaluator(model=None)
        r1, mAP = evaluator.evaluate(
            data_loader=None,
            query=query,
            gallery=gallery,
            cmc_flag=True,
            dataset_name=args.dataset_name,
            external_matrix_manifest=args.manifest,
            external_matrix_mode='replace',
            external_matrix_kind='similarity',
        )
    else:
        distmat = normalize_similarity_to_distance(similarity)
        r1, mAP = evaluate_all(
            query_features=None,
            gallery_features=None,
            distmat=distmat,
            query=query,
            gallery=gallery,
            cmc_flag=True,
            cuhk03=False,
        )

    result_text = (
        f"VPRTempo matrix evaluated with LSTKC metrics\n"
        f"similarity={Path(args.similarity)}\n"
        f"gt={Path(args.gt)}\n"
        f"manifest={args.manifest}\n"
        f"mAP={mAP * 100:.2f}%\n"
        f"R1={r1 * 100:.2f}%\n"
    )
    print(result_text)

    if args.out:
        Path(args.out).write_text(result_text, encoding="utf-8")


if __name__ == "__main__":
    main()
