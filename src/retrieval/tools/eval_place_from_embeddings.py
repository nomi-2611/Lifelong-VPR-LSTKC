from __future__ import absolute_import, print_function

import argparse
import json
import os.path as osp
import sys
import time

import numpy as np

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.datasets.lreid_dataset.datasets as datasets
from src.evaluation.vpr_evaluators import evaluate_vpr_embeddings
from src.retrieval.tools.eval_place_sequence_rerank import _evaluate_stream_candidate_sequence, _load_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate place datasets from exported embeddings only.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset-names', nargs='+', default=['nordland_place', 'robotcar_place', 'tart_place'])
    parser.add_argument('--eval-query-limit', type=int, default=None)
    parser.add_argument('--eval-gallery-limit', type=int, default=None)
    parser.add_argument('--place-vpr-protocol', type=str, default='dataset', choices=['dataset', 'strict_pid'])
    parser.add_argument('--nordland-eval-tolerance', type=int, default=0)
    parser.add_argument('--robotcar-eval-tolerance', type=int, default=1)
    parser.add_argument('--tart-eval-pose-threshold', type=float, default=3.0)
    parser.add_argument('--sequence-rerank', action='store_true',
                        help='Use VPR sequence-consistency reranking instead of plain embedding ranking.')
    parser.add_argument('--sequence-radius', type=int, default=30)
    parser.add_argument('--sequence-weight', type=float, default=1.0)
    parser.add_argument('--candidate-topk', type=int, default=3000)
    parser.add_argument('--query-block-size', type=int, default=16)
    parser.add_argument('--candidate-chunk-size', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.manifest, 'r', encoding='utf-8-sig') as handle:
        manifest = json.load(handle)

    summary = {}
    all_r1 = []
    all_r5 = []
    all_r10 = []
    all_map = []

    for dataset_name in args.dataset_names:
        dataset = datasets.create(dataset_name, args.data_dir)
        query = dataset.query[:args.eval_query_limit] if args.eval_query_limit is not None else dataset.query
        gallery = dataset.gallery[:args.eval_gallery_limit] if args.eval_gallery_limit is not None else dataset.gallery

        item = manifest.get(dataset_name)
        if item is None:
            raise KeyError('Dataset {} missing from manifest {}'.format(dataset_name, args.manifest))

        load_begin = time.time()
        query_embeddings, gallery_embeddings = _load_embeddings(manifest, dataset_name)
        print('Embedding load [{}]: {:.2f}s, query_shape={}, gallery_shape={}'.format(
            dataset_name, time.time() - load_begin, tuple(query_embeddings.shape), tuple(gallery_embeddings.shape)))

        if args.sequence_rerank:
            from src.evaluation.vpr_evaluators import _build_positive_map
            positive_map, protocol_settings = _build_positive_map(
                dataset_name,
                query,
                gallery,
                protocol=args.place_vpr_protocol,
                nordland_tolerance=args.nordland_eval_tolerance,
                robotcar_tolerance=args.robotcar_eval_tolerance,
                tart_pose_threshold=args.tart_eval_pose_threshold,
            )
            rerank_results = _evaluate_stream_candidate_sequence(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                query=query,
                gallery=gallery,
                positive_map=positive_map,
                radii=[int(args.sequence_radius)],
                weights=[float(args.sequence_weight)],
                candidate_topk=args.candidate_topk,
                query_block_size=args.query_block_size,
                candidate_chunk_size=args.candidate_chunk_size,
            )
            rerank_key = 'r{}_w{:.2f}'.format(int(args.sequence_radius), float(args.sequence_weight))
            result = rerank_results[rerank_key]
            r1 = float(result['R@1'])
            mAP = float(result['mAP'])
            r5 = float(result['R@5'])
            r10 = float(result['R@10'])
            print('VPR sequence-rerank results on {}'.format(dataset_name))
            print('Protocol: {}'.format(protocol_settings['protocol_name']))
            print('VPR mAP: {:4.1%}'.format(mAP))
            print('Recall@K:')
            print('  R@{:<4}{:12.1%}'.format(1, r1))
            print('  R@{:<4}{:12.1%}'.format(5, r5))
            print('  R@{:<4}{:12.1%}'.format(10, r10))
        else:
            r1, mAP = evaluate_vpr_embeddings(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                query=query,
                gallery=gallery,
                dataset_name=dataset_name,
                metric=bool(item.get('normalized', False)),
                nordland_tolerance=args.nordland_eval_tolerance,
                robotcar_tolerance=args.robotcar_eval_tolerance,
                tart_pose_threshold=args.tart_eval_pose_threshold,
                protocol=args.place_vpr_protocol,
            )
            r5 = None
            r10 = None

        summary[dataset_name] = {
            'R@1': float(r1),
            'R@5': None if r5 is None else float(r5),
            'R@10': None if r10 is None else float(r10),
            'mAP': float(mAP),
        }
        all_r1.append(float(r1))
        if r5 is not None:
            all_r5.append(float(r5))
        if r10 is not None:
            all_r10.append(float(r10))
        all_map.append(float(mAP))

    if all_r1:
        summary['average'] = {
            'R@1': float(np.mean(all_r1)),
            'R@5': None if not all_r5 else float(np.mean(all_r5)),
            'R@10': None if not all_r10 else float(np.mean(all_r10)),
            'mAP': float(np.mean(all_map)),
        }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
