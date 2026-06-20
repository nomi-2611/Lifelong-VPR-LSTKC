from __future__ import absolute_import, print_function

import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.datasets.lreid_dataset.datasets as datasets
from src.evaluation.vpr_evaluators import _build_positive_map


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a lifelong VPR stage-expert checkpoint pool.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--run-dir', required=True, help='training log directory containing <dataset>_checkpoint.pth.tar')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-names', nargs='+', default=['robotcar_place', 'nordland_place', 'pitts30k_place'])
    parser.add_argument('--feature-kind', choices=['raw', 'projected', 'bn'], default='raw')
    parser.add_argument('--place-vpr-protocol', choices=['dataset', 'strict_pid'], default='dataset')
    parser.add_argument('--nordland-eval-tolerance', type=int, default=0)
    parser.add_argument('--robotcar-eval-tolerance', type=int, default=1)
    parser.add_argument('--tart-eval-pose-threshold', type=float, default=3.0)
    parser.add_argument('--query-block-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-export-if-present', action='store_true')

    parser.add_argument('--MODEL', default='vprtempo_snn', choices=['50x', 'snn_tiny', 'vprtempo_snn'])
    parser.add_argument('--vprtempo-model-path', default=None)
    parser.add_argument('--place-preprocess-mode', default='auto', choices=['auto', 'reid', 'vprtempo'])
    parser.add_argument('--vprtempo-preprocess-cache', action='store_true')
    parser.add_argument('--vprtempo-preprocess-cache-dir', default=None)
    parser.add_argument('--vprtempo-freeze-mode', default='trainable', choices=['frozen', 'trainable'])
    return parser.parse_args()


def _run_export(args, dataset_name, checkpoint_path, output_dir):
    manifest_path = osp.join(output_dir, 'embedding_manifest.json')
    if args.skip_export_if_present and osp.exists(manifest_path):
        return manifest_path
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        osp.join(PROJECT_ROOT, 'tools', 'export_lstkc_place_features.py'),
        '--data-dir', args.data_dir,
        '--dataset-names', dataset_name,
        '--checkpoint', checkpoint_path,
        '--output-dir', output_dir,
        '--feature-kind', args.feature_kind,
        '--MODEL', args.MODEL,
        '--place-preprocess-mode', args.place_preprocess_mode,
        '--vprtempo-freeze-mode', args.vprtempo_freeze_mode,
        '--batch-size', str(args.batch_size),
        '--workers', str(args.workers),
        '--persistent-workers',
        '--prefetch-factor', '8',
    ]
    if args.vprtempo_model_path:
        cmd.extend(['--vprtempo-model-path', args.vprtempo_model_path])
    if args.vprtempo_preprocess_cache:
        cmd.append('--vprtempo-preprocess-cache')
    if args.vprtempo_preprocess_cache_dir:
        cmd.extend(['--vprtempo-preprocess-cache-dir', args.vprtempo_preprocess_cache_dir])
    print('Export expert [{}] from {}'.format(dataset_name, checkpoint_path))
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)
    return manifest_path


def _load_manifest_embeddings(manifest_path, dataset_name):
    with open(manifest_path, 'r', encoding='utf-8-sig') as handle:
        manifest = json.load(handle)
    item = manifest[dataset_name]
    query_embeddings = np.asarray(np.load(item['query_embeddings']), dtype=np.float32)
    gallery_embeddings = np.asarray(np.load(item['gallery_embeddings']), dtype=np.float32)
    return query_embeddings, gallery_embeddings


def _evaluate_embeddings(query_embeddings, gallery_embeddings, positive_map, query_block_size=64, topk=(1, 5, 10)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_tensor = F.normalize(torch.as_tensor(query_embeddings, dtype=torch.float32), dim=1)
    gallery_tensor = F.normalize(torch.as_tensor(gallery_embeddings, dtype=torch.float32), dim=1).to(device)
    max_topk = max(topk)
    recall_hits = {int(k): 0.0 for k in topk}
    ap_scores = []
    valid_queries = 0
    start = time.time()
    for query_start in range(0, query_tensor.size(0), query_block_size):
        query_end = min(query_start + query_block_size, query_tensor.size(0))
        query_block = query_tensor[query_start:query_end].to(device)
        distances = 2.0 - 2.0 * torch.mm(query_block, gallery_tensor.t())
        sorted_indices = torch.argsort(distances, dim=1, descending=False).detach().cpu().numpy()
        for local_index, query_index in enumerate(range(query_start, query_end)):
            positives = np.asarray(positive_map[query_index], dtype=np.int64)
            if positives.size == 0:
                continue
            valid_queries += 1
            ranking = sorted_indices[local_index]
            top_matches = np.isin(ranking[:max_topk], positives, assume_unique=False)
            for k in recall_hits:
                recall_hits[k] += float(top_matches[:k].any())
            positive_mask = np.isin(ranking, positives, assume_unique=False)
            positive_ranks = np.flatnonzero(positive_mask)
            if positive_ranks.size == 0:
                ap_scores.append(0.0)
                continue
            precision = np.arange(1, positive_ranks.size + 1, dtype=np.float32) / (positive_ranks.astype(np.float32) + 1.0)
            ap_scores.append(float(np.mean(precision)))
    if valid_queries == 0:
        raise RuntimeError('No valid queries with positive gallery matches')
    result = {
        'mAP': float(np.mean(ap_scores)) if ap_scores else 0.0,
        'R@1': float(recall_hits[1] / valid_queries),
        'R@5': float(recall_hits[5] / valid_queries),
        'R@10': float(recall_hits[10] / valid_queries),
        'valid_queries': int(valid_queries),
        'eval_seconds': float(time.time() - start),
    }
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = OrderedDict()
    metrics = []
    for dataset_name in args.dataset_names:
        checkpoint_path = osp.join(args.run_dir, '{}_checkpoint.pth.tar'.format(dataset_name))
        if not osp.exists(checkpoint_path):
            raise FileNotFoundError('Missing expert checkpoint: {}'.format(checkpoint_path))
        dataset_output_dir = osp.join(args.output_dir, dataset_name)
        manifest_path = _run_export(args, dataset_name, checkpoint_path, dataset_output_dir)
        query_embeddings, gallery_embeddings = _load_manifest_embeddings(manifest_path, dataset_name)
        dataset = datasets.create(dataset_name, args.data_dir)
        positive_map, protocol_settings = _build_positive_map(
            dataset_name,
            dataset.query,
            dataset.gallery,
            protocol=args.place_vpr_protocol,
            nordland_tolerance=args.nordland_eval_tolerance,
            robotcar_tolerance=args.robotcar_eval_tolerance,
            tart_pose_threshold=args.tart_eval_pose_threshold,
        )
        result = _evaluate_embeddings(
            query_embeddings,
            gallery_embeddings,
            positive_map,
            query_block_size=args.query_block_size,
        )
        result['checkpoint'] = osp.abspath(checkpoint_path).replace('\\', '/')
        result['manifest'] = osp.abspath(manifest_path).replace('\\', '/')
        result['protocol'] = protocol_settings['protocol_name']
        summary[dataset_name] = result
        metrics.append(result)
        print('{} expert: mAP={:.1%}, R@1={:.1%}, R@5={:.1%}, R@10={:.1%}'.format(
            dataset_name, result['mAP'], result['R@1'], result['R@5'], result['R@10']))

    if metrics:
        summary['average'] = {
            'mAP': float(np.mean([item['mAP'] for item in metrics])),
            'R@1': float(np.mean([item['R@1'] for item in metrics])),
            'R@5': float(np.mean([item['R@5'] for item in metrics])),
            'R@10': float(np.mean([item['R@10'] for item in metrics])),
        }
    output_path = osp.join(args.output_dir, 'stage_expert_pool_summary.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print('Stage expert pool summary saved to {}'.format(osp.abspath(output_path).replace('\\', '/')))


if __name__ == '__main__':
    main()
