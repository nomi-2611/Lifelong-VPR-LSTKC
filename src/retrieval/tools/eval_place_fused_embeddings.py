from __future__ import absolute_import, print_function

import argparse
import json
import os.path as osp
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.datasets.lreid_dataset.datasets as datasets
from src.evaluation.vpr_evaluators import _build_positive_map


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VPR by fusing two exported embedding manifests.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--primary-manifest', required=True)
    parser.add_argument('--secondary-manifest', required=True)
    parser.add_argument('--dataset-names', nargs='+', default=['robotcar_place'])
    parser.add_argument('--secondary-weights', type=float, nargs='+', default=[0.5])
    parser.add_argument('--eval-query-limit', type=int, default=None)
    parser.add_argument('--eval-gallery-limit', type=int, default=None)
    parser.add_argument('--place-vpr-protocol', type=str, default='dataset', choices=['dataset', 'strict_pid'])
    parser.add_argument('--nordland-eval-tolerance', type=int, default=0)
    parser.add_argument('--robotcar-eval-tolerance', type=int, default=1)
    parser.add_argument('--tart-eval-pose-threshold', type=float, default=3.0)
    parser.add_argument('--query-block-size', type=int, default=64)
    return parser.parse_args()


def _load_manifest(path):
    with open(path, 'r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def _load_embeddings(manifest, dataset_name):
    item = manifest.get(dataset_name)
    if item is None:
        raise KeyError('Dataset {} missing from manifest'.format(dataset_name))
    query_embeddings = np.asarray(np.load(item['query_embeddings']), dtype=np.float32)
    gallery_embeddings = np.asarray(np.load(item['gallery_embeddings']), dtype=np.float32)
    return query_embeddings, gallery_embeddings


def _to_tensor(array):
    tensor = torch.as_tensor(np.asarray(array, dtype=np.float32)).contiguous()
    return F.normalize(tensor, p=2, dim=1)


def _distance_block(query_block, gallery_device):
    return 2.0 - 2.0 * torch.mm(query_block, gallery_device.t())


def _evaluate_fused(
        primary_query,
        primary_gallery,
        secondary_query,
        secondary_gallery,
        positive_map,
        secondary_weight,
        topk=(1, 5, 10),
        query_block_size=64):
    phase_begin = time.time()
    max_topk = int(max(topk))
    recall_hits = {int(k): 0.0 for k in topk}
    ap_scores = []
    valid_queries = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    primary_gallery_device = primary_gallery.to(device, non_blocking=True)
    secondary_gallery_device = secondary_gallery.to(device, non_blocking=True)
    primary_weight = 1.0 - float(secondary_weight)

    num_queries = primary_query.size(0)
    for query_start in range(0, num_queries, query_block_size):
        query_end = min(query_start + query_block_size, num_queries)
        primary_query_block = primary_query[query_start:query_end].to(device, non_blocking=True)
        secondary_query_block = secondary_query[query_start:query_end].to(device, non_blocking=True)

        primary_dist = _distance_block(primary_query_block, primary_gallery_device)
        secondary_dist = _distance_block(secondary_query_block, secondary_gallery_device)
        fused_dist = primary_dist.mul(primary_weight).add_(secondary_dist, alpha=float(secondary_weight))

        sorted_indices = torch.argsort(fused_dist, dim=1, descending=False)
        topk_indices_cpu = sorted_indices[:, :max_topk].detach().cpu().numpy()
        sorted_indices_cpu = sorted_indices.detach().cpu().numpy()

        for local_index, query_index in enumerate(range(query_start, query_end)):
            positive_indices = np.asarray(positive_map[query_index], dtype=np.int64)
            if positive_indices.size == 0:
                continue
            valid_queries += 1

            topk_matches = np.isin(topk_indices_cpu[local_index], positive_indices, assume_unique=False)
            for k in recall_hits:
                recall_hits[int(k)] += float(topk_matches[:int(k)].any())

            ranking = sorted_indices_cpu[local_index]
            positive_mask = np.isin(ranking, positive_indices, assume_unique=False)
            positive_ranks = np.flatnonzero(positive_mask)
            if positive_ranks.size == 0:
                ap_scores.append(0.0)
                continue
            precision = (np.arange(1, positive_ranks.size + 1, dtype=np.float32) /
                         (positive_ranks.astype(np.float32) + 1.0))
            ap_scores.append(float(np.mean(precision)))

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if valid_queries == 0:
        raise RuntimeError('No valid VPR queries with positives')
    recalls = {int(k): recall_hits[int(k)] / float(valid_queries) for k in topk}
    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    print('Fused eval phase [ranking_metrics]: {:.2f}s, device={}, secondary_weight={:.3f}'.format(
        time.time() - phase_begin, device.type, float(secondary_weight)))
    return recalls, mAP, valid_queries


def main():
    args = parse_args()
    primary_manifest = _load_manifest(args.primary_manifest)
    secondary_manifest = _load_manifest(args.secondary_manifest)

    final_summary = {}
    for dataset_name in args.dataset_names:
        dataset = datasets.create(dataset_name, args.data_dir)
        query = dataset.query[:args.eval_query_limit] if args.eval_query_limit is not None else dataset.query
        gallery = dataset.gallery[:args.eval_gallery_limit] if args.eval_gallery_limit is not None else dataset.gallery

        load_begin = time.time()
        primary_query, primary_gallery = _load_embeddings(primary_manifest, dataset_name)
        secondary_query, secondary_gallery = _load_embeddings(secondary_manifest, dataset_name)
        print('Embedding load [{}]: {:.2f}s, primary_q={}, primary_g={}, secondary_q={}, secondary_g={}'.format(
            dataset_name,
            time.time() - load_begin,
            tuple(primary_query.shape),
            tuple(primary_gallery.shape),
            tuple(secondary_query.shape),
            tuple(secondary_gallery.shape),
        ))

        if primary_query.shape[0] != len(query) or secondary_query.shape[0] != len(query):
            raise ValueError('Query embedding count does not match dataset query size')
        if primary_gallery.shape[0] != len(gallery) or secondary_gallery.shape[0] != len(gallery):
            raise ValueError('Gallery embedding count does not match dataset gallery size')

        positive_map, protocol_settings = _build_positive_map(
            dataset_name,
            query,
            gallery,
            protocol=args.place_vpr_protocol,
            nordland_tolerance=args.nordland_eval_tolerance,
            robotcar_tolerance=args.robotcar_eval_tolerance,
            tart_pose_threshold=args.tart_eval_pose_threshold,
        )
        print('Protocol [{}]: {}, valid_positive_queries={}'.format(
            dataset_name,
            protocol_settings['protocol_name'],
            int(sum(1 for positives in positive_map if positives.size > 0)),
        ))

        primary_query_tensor = _to_tensor(primary_query)
        primary_gallery_tensor = _to_tensor(primary_gallery)
        secondary_query_tensor = _to_tensor(secondary_query)
        secondary_gallery_tensor = _to_tensor(secondary_gallery)

        dataset_summary = {}
        for secondary_weight in args.secondary_weights:
            recalls, mAP, valid_queries = _evaluate_fused(
                primary_query_tensor,
                primary_gallery_tensor,
                secondary_query_tensor,
                secondary_gallery_tensor,
                positive_map,
                secondary_weight=float(secondary_weight),
                query_block_size=args.query_block_size,
            )
            key = 'secondary_weight_{:.3f}'.format(float(secondary_weight))
            dataset_summary[key] = {
                'mAP': float(mAP),
                'R@1': float(recalls[1]),
                'R@5': float(recalls[5]),
                'R@10': float(recalls[10]),
                'valid_queries': int(valid_queries),
            }
            print('{} weight={:.3f}: mAP={:.1%}, R@1={:.1%}, R@5={:.1%}, R@10={:.1%}'.format(
                dataset_name,
                float(secondary_weight),
                float(mAP),
                float(recalls[1]),
                float(recalls[5]),
                float(recalls[10]),
            ))
        final_summary[dataset_name] = dataset_summary

    print(json.dumps(final_summary, indent=2))


if __name__ == '__main__':
    main()
