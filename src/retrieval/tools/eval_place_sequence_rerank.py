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
    parser = argparse.ArgumentParser(
        description='Evaluate place embeddings with VPR sequence-consistency reranking.'
    )
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset-names', nargs='+', default=['robotcar_place', 'nordland_place'])
    parser.add_argument('--sequence-radii', type=int, nargs='+', default=[0, 1, 2, 3, 5, 10])
    parser.add_argument('--sequence-weights', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--query-block-size', type=int, default=64)
    parser.add_argument('--candidate-topk', type=int, default=200,
                        help='Only rerank/evaluate candidates from base top-k for faster large-scale sweeps. '
                             'Use 0 for exact full-gallery evaluation.')
    parser.add_argument('--stream-candidate-rerank', action='store_true',
                        help='Avoid materializing full distance matrices; rerank base top-k candidates in query blocks.')
    parser.add_argument('--candidate-chunk-size', type=int, default=64,
                        help='Candidate chunk size used by stream sequence reranking to control GPU memory.')
    parser.add_argument('--place-vpr-protocol', type=str, default='dataset', choices=['dataset', 'strict_pid'])
    parser.add_argument('--nordland-eval-tolerance', type=int, default=0)
    parser.add_argument('--robotcar-eval-tolerance', type=int, default=1)
    parser.add_argument('--tart-eval-pose-threshold', type=float, default=3.0)
    parser.add_argument('--output-json', type=str, default=None)
    return parser.parse_args()


def _load_manifest(path):
    with open(path, 'r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def _load_embeddings(manifest, dataset_name):
    item = manifest.get(dataset_name)
    if item is None:
        raise KeyError('Dataset {} missing from {}'.format(dataset_name, manifest))
    query_embeddings = np.asarray(np.load(item['query_embeddings']), dtype=np.float32)
    gallery_embeddings = np.asarray(np.load(item['gallery_embeddings']), dtype=np.float32)
    return query_embeddings, gallery_embeddings


def _to_normalized_tensor(array):
    tensor = torch.as_tensor(np.asarray(array, dtype=np.float32)).contiguous()
    return F.normalize(tensor, p=2, dim=1)


def _distance_matrix(query_embeddings, gallery_embeddings):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_tensor = _to_normalized_tensor(query_embeddings).to(device, non_blocking=True)
    gallery_tensor = _to_normalized_tensor(gallery_embeddings).to(device, non_blocking=True)
    dist = 2.0 - 2.0 * torch.mm(query_tensor, gallery_tensor.t())
    dist = dist.clamp_min_(0.0).detach().cpu().numpy().astype(np.float32)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return dist


def _groups_by_cam(entries):
    groups = []
    start = 0
    while start < len(entries):
        camid = entries[start][2]
        end = start + 1
        while end < len(entries) and entries[end][2] == camid:
            end += 1
        groups.append((start, end, camid))
        start = end
    return groups


def _group_bounds_by_index(entries):
    starts = np.zeros((len(entries),), dtype=np.int64)
    ends = np.zeros((len(entries),), dtype=np.int64)
    for start, end, _ in _groups_by_cam(entries):
        starts[start:end] = int(start)
        ends[start:end] = int(end)
    return starts, ends


def _sequence_average_distance(base_dist, query, gallery, radius):
    radius = int(radius)
    if radius <= 0:
        return base_dist.copy()

    seq_sum = np.zeros_like(base_dist, dtype=np.float32)
    seq_count = np.zeros_like(base_dist, dtype=np.float32)
    query_groups = _groups_by_cam(query)
    gallery_groups = _groups_by_cam(gallery)

    for query_start, query_end, _ in query_groups:
        query_len = query_end - query_start
        for gallery_start, gallery_end, _ in gallery_groups:
            gallery_len = gallery_end - gallery_start
            block = base_dist[query_start:query_end, gallery_start:gallery_end]
            block_sum = np.zeros_like(block, dtype=np.float32)
            block_count = np.zeros_like(block, dtype=np.float32)

            for offset in range(-radius, radius + 1):
                if offset >= 0:
                    src_q = slice(offset, query_len)
                    dst_q = slice(0, query_len - offset)
                    src_g = slice(offset, gallery_len)
                    dst_g = slice(0, gallery_len - offset)
                else:
                    shift = -offset
                    src_q = slice(0, query_len - shift)
                    dst_q = slice(shift, query_len)
                    src_g = slice(0, gallery_len - shift)
                    dst_g = slice(shift, gallery_len)

                if src_q.stop <= src_q.start or src_g.stop <= src_g.start:
                    continue
                shifted = block[src_q, src_g]
                block_sum[dst_q, dst_g] += shifted
                block_count[dst_q, dst_g] += 1.0

            valid = block_count > 0
            block_seq = block.copy()
            block_seq[valid] = block_sum[valid] / block_count[valid]
            seq_sum[query_start:query_end, gallery_start:gallery_end] = block_seq
            seq_count[query_start:query_end, gallery_start:gallery_end] = 1.0

    missing = seq_count <= 0
    seq_sum[missing] = base_dist[missing]
    return seq_sum


def _evaluate_distance_matrix(distmat, positive_map, topk=(1, 5, 10)):
    max_topk = int(max(topk))
    recall_hits = {int(k): 0.0 for k in topk}
    ap_scores = []
    valid_queries = 0

    sorted_indices = np.argsort(distmat, axis=1)
    topk_indices = sorted_indices[:, :max_topk]

    for query_index, positive_indices in enumerate(positive_map):
        positive_indices = np.asarray(positive_indices, dtype=np.int64)
        if positive_indices.size == 0:
            continue
        valid_queries += 1

        topk_matches = np.isin(topk_indices[query_index], positive_indices, assume_unique=False)
        for k in recall_hits:
            recall_hits[int(k)] += float(topk_matches[:int(k)].any())

        ranking = sorted_indices[query_index]
        positive_mask = np.isin(ranking, positive_indices, assume_unique=False)
        positive_ranks = np.flatnonzero(positive_mask)
        if positive_ranks.size == 0:
            ap_scores.append(0.0)
            continue
        precision = (
            np.arange(1, positive_ranks.size + 1, dtype=np.float32) /
            (positive_ranks.astype(np.float32) + 1.0)
        )
        ap_scores.append(float(np.mean(precision)))

    if valid_queries == 0:
        raise RuntimeError('No valid VPR queries with positives')
    recalls = {int(k): recall_hits[int(k)] / float(valid_queries) for k in topk}
    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    return recalls, mAP, valid_queries


def _evaluate_candidate_distance_matrix(base_dist, rerank_dist, positive_map, candidate_topk=200, topk=(1, 5, 10)):
    candidate_topk = int(candidate_topk)
    if candidate_topk <= 0 or candidate_topk >= base_dist.shape[1]:
        return _evaluate_distance_matrix(rerank_dist, positive_map, topk=topk)

    max_topk = int(max(topk))
    candidate_topk = max(candidate_topk, max_topk)
    candidate_indices = np.argpartition(base_dist, kth=candidate_topk - 1, axis=1)[:, :candidate_topk]
    candidate_scores = np.take_along_axis(rerank_dist, candidate_indices, axis=1)
    candidate_order = np.argsort(candidate_scores, axis=1)
    ranked_candidates = np.take_along_axis(candidate_indices, candidate_order, axis=1)

    recall_hits = {int(k): 0.0 for k in topk}
    ap_scores = []
    valid_queries = 0

    for query_index, positive_indices in enumerate(positive_map):
        positive_indices = np.asarray(positive_indices, dtype=np.int64)
        if positive_indices.size == 0:
            continue
        valid_queries += 1

        ranking = ranked_candidates[query_index]
        topk_matches = np.isin(ranking[:max_topk], positive_indices, assume_unique=False)
        for k in recall_hits:
            recall_hits[int(k)] += float(topk_matches[:int(k)].any())

        positive_mask = np.isin(ranking, positive_indices, assume_unique=False)
        positive_ranks = np.flatnonzero(positive_mask)
        if positive_ranks.size == 0:
            ap_scores.append(0.0)
            continue
        precision = (
            np.arange(1, positive_ranks.size + 1, dtype=np.float32) /
            (positive_ranks.astype(np.float32) + 1.0)
        )
        ap_scores.append(float(np.mean(precision)))

    if valid_queries == 0:
        raise RuntimeError('No valid VPR queries with positives')
    recalls = {int(k): recall_hits[int(k)] / float(valid_queries) for k in topk}
    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    return recalls, mAP, valid_queries


def _evaluate_stream_candidate_sequence(
        query_embeddings,
        gallery_embeddings,
        query,
        gallery,
        positive_map,
        radii,
        weights,
        candidate_topk=1000,
        query_block_size=64,
        candidate_chunk_size=64,
        topk=(1, 5, 10)):
    begin = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_tensor = _to_normalized_tensor(query_embeddings).to(device, non_blocking=True)
    gallery_tensor = _to_normalized_tensor(gallery_embeddings).to(device, non_blocking=True)
    num_queries = int(query_tensor.size(0))
    num_gallery = int(gallery_tensor.size(0))
    candidate_topk = min(max(int(candidate_topk), int(max(topk))), num_gallery)

    query_group_starts, query_group_ends = _group_bounds_by_index(query)
    gallery_group_starts, gallery_group_ends = _group_bounds_by_index(gallery)
    query_indices_all = torch.arange(num_queries, device=device, dtype=torch.long)
    candidate_chunk_size = max(1, int(candidate_chunk_size))

    result_state = {}
    for radius in radii:
        for weight in weights:
            key = 'r{}_w{:.2f}'.format(int(radius), float(weight))
            result_state[key] = {
                'recall_hits': {int(k): 0.0 for k in topk},
                'ap_scores': [],
                'valid_queries': 0,
            }

    max_topk = int(max(topk))
    for query_start in range(0, num_queries, int(query_block_size)):
        query_end = min(query_start + int(query_block_size), num_queries)
        query_block = query_tensor[query_start:query_end]
        base_similarity, candidate_indices = torch.topk(
            torch.mm(query_block, gallery_tensor.t()),
            k=candidate_topk,
            dim=1,
            largest=True,
            sorted=True,
        )
        block_query_indices = query_indices_all[query_start:query_end]
        block_size = int(query_end - query_start)
        candidate_indices_cpu = candidate_indices.detach().cpu().numpy()
        base_similarity_cpu = base_similarity.detach().cpu().numpy().astype(np.float32)
        query_group_start_tensor = torch.as_tensor(
            query_group_starts[query_start:query_end],
            device=device,
            dtype=torch.long,
        )
        query_group_end_tensor = torch.as_tensor(
            query_group_ends[query_start:query_end],
            device=device,
            dtype=torch.long,
        )
        gallery_start_full = torch.as_tensor(
            gallery_group_starts[candidate_indices_cpu.reshape(-1)].reshape(block_size, candidate_topk),
            device=device,
            dtype=torch.long,
        )
        gallery_end_full = torch.as_tensor(
            gallery_group_ends[candidate_indices_cpu.reshape(-1)].reshape(block_size, candidate_topk),
            device=device,
            dtype=torch.long,
        )

        seq_similarity_by_radius = {}
        for radius in radii:
            radius = int(radius)
            if radius <= 0:
                seq_similarity_by_radius[radius] = base_similarity_cpu
                continue

            score_sum = torch.zeros_like(base_similarity)
            score_count = torch.zeros_like(base_similarity)
            candidate_indices_long = candidate_indices.long()
            for offset in range(-radius, radius + 1):
                shifted_query = block_query_indices + int(offset)
                valid_query = (
                    (shifted_query >= query_group_start_tensor) &
                    (shifted_query < query_group_end_tensor)
                )
                safe_query = shifted_query.clamp(0, num_queries - 1)
                shifted_query_features = query_tensor[safe_query]

                for cand_start in range(0, candidate_topk, candidate_chunk_size):
                    cand_end = min(cand_start + candidate_chunk_size, candidate_topk)
                    shifted_gallery = candidate_indices_long[:, cand_start:cand_end] + int(offset)
                    valid_gallery = (
                        (shifted_gallery >= gallery_start_full[:, cand_start:cand_end]) &
                        (shifted_gallery < gallery_end_full[:, cand_start:cand_end])
                    )
                    valid = valid_query.reshape(-1, 1) & valid_gallery
                    if not bool(valid.any().item()):
                        continue
                    safe_gallery = shifted_gallery.clamp(0, num_gallery - 1)
                    shifted_gallery_features = gallery_tensor[safe_gallery.reshape(-1)].reshape(
                        block_size, cand_end - cand_start, -1
                    )
                    shifted_similarity = (
                        shifted_gallery_features * shifted_query_features.unsqueeze(1)
                    ).sum(dim=2)
                    shifted_similarity = torch.where(
                        valid,
                        shifted_similarity,
                        torch.zeros_like(shifted_similarity),
                    )
                    score_sum[:, cand_start:cand_end] += shifted_similarity
                    score_count[:, cand_start:cand_end] += valid.float()

            seq_similarity = torch.where(score_count > 0, score_sum / score_count.clamp_min(1.0), base_similarity)
            seq_similarity_by_radius[radius] = seq_similarity.detach().cpu().numpy().astype(np.float32)

        for radius in radii:
            radius = int(radius)
            seq_similarity = seq_similarity_by_radius[radius]
            for weight in weights:
                weight = float(weight)
                key = 'r{}_w{:.2f}'.format(radius, weight)
                fused_similarity = base_similarity_cpu * (1.0 - weight) + seq_similarity * weight
                order = np.argsort(-fused_similarity, axis=1)
                ranked_candidates = np.take_along_axis(candidate_indices_cpu, order, axis=1)
                state = result_state[key]

                for local_index, query_index in enumerate(range(query_start, query_end)):
                    positive_indices = np.asarray(positive_map[query_index], dtype=np.int64)
                    if positive_indices.size == 0:
                        continue
                    state['valid_queries'] += 1
                    ranking = ranked_candidates[local_index]
                    topk_matches = np.isin(ranking[:max_topk], positive_indices, assume_unique=False)
                    for k in state['recall_hits']:
                        state['recall_hits'][int(k)] += float(topk_matches[:int(k)].any())

                    positive_mask = np.isin(ranking, positive_indices, assume_unique=False)
                    positive_ranks = np.flatnonzero(positive_mask)
                    if positive_ranks.size == 0:
                        state['ap_scores'].append(0.0)
                    else:
                        precision = (
                            np.arange(1, positive_ranks.size + 1, dtype=np.float32) /
                            (positive_ranks.astype(np.float32) + 1.0)
                        )
                        state['ap_scores'].append(float(np.mean(precision)))

        if ((query_end // int(query_block_size)) % 50 == 0) or query_end == num_queries:
            print('stream rerank progress: {}/{} queries, elapsed={:.1f}s'.format(
                int(query_end), int(num_queries), time.time() - begin))

    final_results = {}
    for key, state in result_state.items():
        valid_queries = int(state['valid_queries'])
        if valid_queries == 0:
            continue
        recalls = {
            int(k): float(state['recall_hits'][int(k)]) / float(valid_queries)
            for k in topk
        }
        mAP = float(np.mean(state['ap_scores'])) if state['ap_scores'] else 0.0
        final_results[key] = {
            'mAP': mAP,
            'R@1': float(recalls[1]),
            'R@5': float(recalls[5]),
            'R@10': float(recalls[10]),
            'valid_queries': valid_queries,
            'candidate_topk': int(candidate_topk),
            'candidate_metric_note': 'R@K/mAP computed after reranking base top-k candidates',
        }

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('stream rerank total: {:.2f}s, device={}, candidate_topk={}'.format(
        time.time() - begin, device.type, int(candidate_topk)))
    return final_results


def _evaluate_dataset(args, manifest, dataset_name):
    dataset = datasets.create(dataset_name, args.data_dir)
    query = dataset.query
    gallery = dataset.gallery
    query_embeddings, gallery_embeddings = _load_embeddings(manifest, dataset_name)
    if query_embeddings.shape[0] != len(query):
        raise ValueError('{} query embedding count mismatch'.format(dataset_name))
    if gallery_embeddings.shape[0] != len(gallery):
        raise ValueError('{} gallery embedding count mismatch'.format(dataset_name))

    positive_map, protocol_settings = _build_positive_map(
        dataset_name,
        query,
        gallery,
        protocol=args.place_vpr_protocol,
        nordland_tolerance=args.nordland_eval_tolerance,
        robotcar_tolerance=args.robotcar_eval_tolerance,
        tart_pose_threshold=args.tart_eval_pose_threshold,
    )

    dataset_results = {}
    best_key = None
    best_r1 = -1.0
    best_map = -1.0

    if args.stream_candidate_rerank:
        dataset_results = _evaluate_stream_candidate_sequence(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            query=query,
            gallery=gallery,
            positive_map=positive_map,
            radii=[int(radius) for radius in args.sequence_radii],
            weights=[float(weight) for weight in args.sequence_weights],
            candidate_topk=args.candidate_topk,
            query_block_size=args.query_block_size,
            candidate_chunk_size=args.candidate_chunk_size,
        )
        for key, item in sorted(dataset_results.items()):
            print('{} {}: mAP={:.1%}, R@1={:.1%}, R@5={:.1%}, R@10={:.1%}'.format(
                dataset_name, key, item['mAP'], item['R@1'], item['R@5'], item['R@10']))
            if item['R@1'] > best_r1 or (item['R@1'] == best_r1 and item['mAP'] > best_map):
                best_key = key
                best_r1 = item['R@1']
                best_map = item['mAP']
        return {
            'best_key': best_key,
            'best': dataset_results[best_key],
            'all': dataset_results,
        }

    begin = time.time()
    base_dist = _distance_matrix(query_embeddings, gallery_embeddings)
    print('{} base distance: {:.2f}s, shape={}'.format(
        dataset_name, time.time() - begin, tuple(base_dist.shape)))

    for radius in args.sequence_radii:
        seq_begin = time.time()
        seq_dist = _sequence_average_distance(base_dist, query, gallery, radius=int(radius))
        print('{} sequence radius {}: {:.2f}s'.format(
            dataset_name, int(radius), time.time() - seq_begin))
        for sequence_weight in args.sequence_weights:
            sequence_weight = float(sequence_weight)
            fused_dist = base_dist * (1.0 - sequence_weight) + seq_dist * sequence_weight
            recalls, mAP, valid_queries = _evaluate_candidate_distance_matrix(
                base_dist,
                fused_dist,
                positive_map,
                candidate_topk=args.candidate_topk,
            )
            key = 'r{}_w{:.2f}'.format(int(radius), sequence_weight)
            dataset_results[key] = {
                'mAP': float(mAP),
                'R@1': float(recalls[1]),
                'R@5': float(recalls[5]),
                'R@10': float(recalls[10]),
                'valid_queries': int(valid_queries),
                'protocol': protocol_settings['protocol_name'],
            }
            print('{} {}: mAP={:.1%}, R@1={:.1%}, R@5={:.1%}, R@10={:.1%}'.format(
                dataset_name, key, float(mAP), float(recalls[1]), float(recalls[5]), float(recalls[10])))
            if float(recalls[1]) > best_r1 or (float(recalls[1]) == best_r1 and float(mAP) > best_map):
                best_key = key
                best_r1 = float(recalls[1])
                best_map = float(mAP)

    return {
        'best_key': best_key,
        'best': dataset_results[best_key],
        'all': dataset_results,
    }


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    summary = {}
    best_r1_values = []
    best_map_values = []

    for dataset_name in args.dataset_names:
        result = _evaluate_dataset(args, manifest, dataset_name)
        summary[dataset_name] = result
        best_r1_values.append(result['best']['R@1'])
        best_map_values.append(result['best']['mAP'])

    if best_r1_values:
        summary['average_best'] = {
            'mAP': float(np.mean(best_map_values)),
            'R@1': float(np.mean(best_r1_values)),
        }
    print(json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)


if __name__ == '__main__':
    main()
