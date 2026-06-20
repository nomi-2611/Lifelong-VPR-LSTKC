from __future__ import print_function, absolute_import

import argparse
import json
import os
import os.path as osp
import sys

from torch.backends import cudnn

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.lreid_dataset import datasets
from src.datasets.lreid_dataset.datasets.get_data_loaders import get_test_loader
from src.evaluation.evaluators import Evaluator
from src.models.reid_models.layers import DataParallel
from src.models.reid_models.resnet import make_model


def main():
    parser = argparse.ArgumentParser(description="Smoke test for place-recognition dataset support in LSTKC")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--logs-dir", type=str, required=True)
    parser.add_argument("--query-limit", type=int, default=128)
    parser.add_argument("--gallery-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dataset-names", nargs="+", default=["nordland_place", "robotcar_place", "tart_place"])
    parser.add_argument("--external-embedding-manifest", type=str, default=None)
    parser.add_argument("--external-matrix-manifest", type=str, default=None)
    parser.add_argument("--external-matrix-mode", type=str, default="replace", choices=["replace", "fuse"])
    parser.add_argument("--external-matrix-weight", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    cudnn.benchmark = False

    class SmokeArgs(object):
        MODEL = "50x"

    model = make_model(SmokeArgs(), num_class=1, camera_num=0, view_num=0)
    model.cuda()
    model = DataParallel(model)
    evaluator = Evaluator(model)

    results = []

    for dataset_name in args.dataset_names:
        dataset = datasets.create(dataset_name, args.data_dir)
        query = dataset.query[:args.query_limit]
        gallery = dataset.gallery[:args.gallery_limit]
        testset = list({*query, *gallery})
        test_loader = get_test_loader(dataset, 256, 128, args.batch_size, args.workers, testset=testset)

        r1, mAP = evaluator.evaluate(
            test_loader,
            query,
            gallery,
            cmc_flag=True,
            dataset_name=dataset_name,
            task_type="place",
            external_embedding_manifest=args.external_embedding_manifest,
            external_matrix_manifest=args.external_matrix_manifest,
            external_matrix_mode=args.external_matrix_mode,
            external_matrix_weight=args.external_matrix_weight,
        )
        results.append({
            "dataset": dataset_name,
            "query_count": len(query),
            "gallery_count": len(gallery),
            "mAP": float(mAP),
            "R1": float(r1),
        })

    summary_path = osp.join(args.logs_dir, "place_smoke_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved smoke-test summary to {}".format(summary_path))
    for item in results:
        print(
            "{}: queries={} gallery={} mAP={:.4f} R1={:.4f}".format(
                item["dataset"], item["query_count"], item["gallery_count"], item["mAP"], item["R1"]
            )
        )


if __name__ == "__main__":
    main()
