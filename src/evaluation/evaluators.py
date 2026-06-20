from __future__ import print_function, absolute_import
import json
import os
import os.path as osp
import re
import time
from collections import OrderedDict
import numpy as np
import torch

from src.evaluation.evaluation_metrics import cmc, mean_ap, mean_ap_cuhk03
from src.models.feature_extraction import extract_cnn_feature
from src.utils.reid_utils.meters import AverageMeter
from src.utils.reid_utils.rerank import re_ranking
from torch.nn import functional as F


def _normalize_numpy_matrix(matrix):
    matrix = matrix.astype(np.float32)
    min_v = float(matrix.min())
    max_v = float(matrix.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(matrix, dtype=np.float32)
    return (matrix - min_v) / (max_v - min_v)


def _normalize_numpy_matrix_per_row(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, 1e-12)
    return (matrix - row_min) / denom


def _rank_to_rrf_score(matrix, k=60):
    indices = np.argsort(matrix, axis=1)
    ranks = np.empty_like(indices, dtype=np.int32)
    row_ids = np.arange(indices.shape[0])[:, None]
    ranks[row_ids, indices] = np.arange(indices.shape[1])[None, :]
    return 1.0 / (k + ranks.astype(np.float32) + 1.0)


def _compute_row_confidence_from_distance(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError('Distance matrix must be 2D, got {}'.format(matrix.ndim))
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    sorted_rows = np.sort(matrix, axis=1)
    best = sorted_rows[:, 0]
    if matrix.shape[1] == 1:
        second_best = np.ones_like(best, dtype=np.float32)
    else:
        second_best = sorted_rows[:, 1]
    return (second_best - best).astype(np.float32)


def _compute_row_confidence_from_similarity(matrix, mode='gap',
                                            gap_weight=1.0,
                                            entropy_weight=1.0,
                                            maxsim_weight=1.0,
                                            entropy_softmax_temp=10.0):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError('Similarity matrix must be 2D, got {}'.format(matrix.ndim))
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0],), dtype=np.float32)

    sorted_scores = np.sort(matrix, axis=1)[:, ::-1]
    top1 = sorted_scores[:, 0]
    top2 = sorted_scores[:, 1] if matrix.shape[1] > 1 else np.zeros_like(top1)
    gap = np.maximum(top1 - top2, 0.0)

    logits = matrix * float(entropy_softmax_temp)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    entropy_norm = entropy / np.log(max(matrix.shape[1], 2))
    entropy_certainty = 1.0 - entropy_norm.astype(np.float32)

    if mode == 'gap':
        return (gap_weight * gap).astype(np.float32)
    if mode == 'gap_entropy':
        return (gap_weight * gap + entropy_weight * entropy_certainty).astype(np.float32)
    if mode == 'gap_entropy_max':
        return (gap_weight * gap + entropy_weight * entropy_certainty + maxsim_weight * top1).astype(np.float32)
    raise ValueError('Unsupported adaptive confidence mode: {}'.format(mode))


def _sigmoid_numpy(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _fuse_distance_matrices(primary_dist, secondary_dist, weight=0.5, strategy='linear', rrf_k=60,
                            adaptive_temp=10.0,
                            adaptive_confidence_mode='gap',
                            adaptive_alpha_min=0.0,
                            adaptive_alpha_max=1.0,
                            adaptive_gap_weight=1.0,
                            adaptive_entropy_weight=1.0,
                            adaptive_maxsim_weight=1.0,
                            adaptive_entropy_softmax_temp=10.0,
                            return_details=False):
    primary_dist = _to_numpy_distmat(primary_dist)
    secondary_dist = _to_numpy_distmat(secondary_dist)
    details = {
        'strategy': strategy,
        'weight': float(weight),
    }
    if strategy == 'linear':
        primary_norm = _normalize_numpy_matrix(primary_dist)
        secondary_norm = _normalize_numpy_matrix(secondary_dist)
        fused = weight * primary_norm + (1.0 - weight) * secondary_norm
        return (fused, details) if return_details else fused
    if strategy == 'rowwise_linear':
        primary_norm = _normalize_numpy_matrix_per_row(primary_dist)
        secondary_norm = _normalize_numpy_matrix_per_row(secondary_dist)
        fused = weight * primary_norm + (1.0 - weight) * secondary_norm
        return (fused, details) if return_details else fused
    if strategy == 'rrf':
        primary_score = _rank_to_rrf_score(primary_dist, k=rrf_k)
        secondary_score = _rank_to_rrf_score(secondary_dist, k=rrf_k)
        fused_score = weight * primary_score + (1.0 - weight) * secondary_score
        fused = (1.0 - _normalize_numpy_matrix_per_row(fused_score)).astype(np.float32)
        return (fused, details) if return_details else fused
    if strategy == 'confidence_adaptive':
        primary_norm = _normalize_numpy_matrix_per_row(primary_dist)
        secondary_norm = _normalize_numpy_matrix_per_row(secondary_dist)
        primary_sim = 1.0 - primary_norm
        secondary_sim = 1.0 - secondary_norm
        conf_primary = _compute_row_confidence_from_similarity(
            primary_sim,
            mode=adaptive_confidence_mode,
            gap_weight=adaptive_gap_weight,
            entropy_weight=adaptive_entropy_weight,
            maxsim_weight=adaptive_maxsim_weight,
            entropy_softmax_temp=adaptive_entropy_softmax_temp,
        )
        conf_secondary = _compute_row_confidence_from_similarity(
            secondary_sim,
            mode=adaptive_confidence_mode,
            gap_weight=adaptive_gap_weight,
            entropy_weight=adaptive_entropy_weight,
            maxsim_weight=adaptive_maxsim_weight,
            entropy_softmax_temp=adaptive_entropy_softmax_temp,
        )
        alpha_base = _sigmoid_numpy(adaptive_temp * (conf_primary - conf_secondary)).astype(np.float32)
        alpha = adaptive_alpha_min + (adaptive_alpha_max - adaptive_alpha_min) * alpha_base
        fused = alpha[:, None] * primary_norm + (1.0 - alpha)[:, None] * secondary_norm
        details.update({
            'adaptive_temp': float(adaptive_temp),
            'adaptive_confidence_mode': adaptive_confidence_mode,
            'adaptive_alpha_min_bound': float(adaptive_alpha_min),
            'adaptive_alpha_max_bound': float(adaptive_alpha_max),
            'alpha_mean': float(alpha.mean()),
            'alpha_min': float(alpha.min()),
            'alpha_max': float(alpha.max()),
            'conf_primary_mean': float(conf_primary.mean()),
            'conf_secondary_mean': float(conf_secondary.mean()),
        })
        return (fused.astype(np.float32), details) if return_details else fused.astype(np.float32)
    raise ValueError('Unsupported fusion_strategy: {}'.format(strategy))


def _load_external_matrix(matrix_path, expected_shape=None, matrix_kind='distance'):
    matrix = np.load(matrix_path)
    if isinstance(matrix, np.lib.npyio.NpzFile):
        if 'arr_0' in matrix.files:
            matrix = matrix['arr_0']
        elif len(matrix.files) == 1:
            matrix = matrix[matrix.files[0]]
        else:
            raise KeyError('NPZ matrix file must contain a single array or arr_0')
    matrix = np.asarray(matrix, dtype=np.float32)
    if expected_shape is not None and tuple(matrix.shape) != tuple(expected_shape):
        raise ValueError('External matrix shape {} does not match expected shape {}'.format(matrix.shape, expected_shape))
    if matrix_kind == 'similarity':
        matrix = 1.0 - _normalize_numpy_matrix(matrix)
    return matrix


def _resolve_external_matrix_path(base_dir, dataset_name, suffix):
    if base_dir is None:
        return None
    candidate = os.path.join(base_dir, '{}_{}.npy'.format(dataset_name, suffix))
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(base_dir, '{}_{}.npz'.format(dataset_name, suffix))
    if os.path.exists(candidate):
        return candidate
    return None


def _load_external_matrix_manifest(manifest_path, dataset_name):
    if manifest_path is None:
        return None
    # Accept both plain UTF-8 and UTF-8-with-BOM manifests generated on Windows.
    with open(manifest_path, 'r', encoding='utf-8-sig') as f:
        manifest = json.load(f)
    item = manifest.get(dataset_name)
    if item is None:
        return None
    return {
        'qg': item.get('qg'),
        'qq': item.get('qq'),
        'gg': item.get('gg'),
        'kind': item.get('kind'),
    }


def _load_external_embedding_manifest(manifest_path, dataset_name):
    if manifest_path is None:
        return None
    # Accept both plain UTF-8 and UTF-8-with-BOM manifests generated on Windows.
    with open(manifest_path, 'r', encoding='utf-8-sig') as f:
        manifest = json.load(f)
    item = manifest.get(dataset_name)
    if item is None:
        return None
    if item.get('query_embeddings') is None or item.get('gallery_embeddings') is None:
        return None
    return {
        'query_embeddings': item.get('query_embeddings'),
        'gallery_embeddings': item.get('gallery_embeddings'),
        'metric': item.get('metric', 'cosine'),
        'normalized': bool(item.get('normalized', False)),
    }


def _load_external_embeddings(query_path, gallery_path, expected_query=None, expected_gallery=None):
    query_embeddings = np.asarray(np.load(query_path), dtype=np.float32)
    gallery_embeddings = np.asarray(np.load(gallery_path), dtype=np.float32)
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError('External embeddings must be 2D arrays')
    if expected_query is not None:
        if query_embeddings.shape[0] < expected_query:
            raise ValueError('Query embedding count {} is smaller than expected {}'.format(query_embeddings.shape[0], expected_query))
        if query_embeddings.shape[0] > expected_query:
            query_embeddings = query_embeddings[:expected_query]
    if expected_gallery is not None:
        if gallery_embeddings.shape[0] < expected_gallery:
            raise ValueError('Gallery embedding count {} is smaller than expected {}'.format(gallery_embeddings.shape[0], expected_gallery))
        if gallery_embeddings.shape[0] > expected_gallery:
            gallery_embeddings = gallery_embeddings[:expected_gallery]
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError('Embedding dimensions do not match: {} vs {}'.format(query_embeddings.shape[1], gallery_embeddings.shape[1]))
    return query_embeddings, gallery_embeddings


def _pairwise_distance_from_embeddings(query_embeddings, gallery_embeddings, metric='cosine', normalized=False):
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    gallery_embeddings = np.asarray(gallery_embeddings, dtype=np.float32)
    if metric == 'cosine':
        if not normalized:
            query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
            gallery_norm = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-12
            query_embeddings = query_embeddings / query_norm
            gallery_embeddings = gallery_embeddings / gallery_norm
        similarity = np.matmul(query_embeddings, gallery_embeddings.T)
        return (1.0 - similarity).astype(np.float32)
    if metric == 'euclidean':
        q_sq = np.sum(np.square(query_embeddings), axis=1, keepdims=True)
        g_sq = np.sum(np.square(gallery_embeddings), axis=1, keepdims=True).T
        distmat = q_sq + g_sq - 2.0 * np.matmul(query_embeddings, gallery_embeddings.T)
        return np.maximum(distmat, 0.0).astype(np.float32)
    raise ValueError('Unsupported external embedding metric: {}'.format(metric))


def _to_numpy_distmat(distmat):
    if torch.is_tensor(distmat):
        return distmat.detach().cpu().numpy().astype(np.float32)
    return np.asarray(distmat, dtype=np.float32)

def extract_features(model, data_loader,training_phase=None):
    model.eval()
    try:
        model = model.module
    except:
        model=model
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domians) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs,training_phase=training_phase)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels

def extract_features_print(model, data_loader,training_phase=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domians) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs,training_phase=training_phase)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels




def pairwise_distance(features, query=None, gallery=None, metric=False):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not False:
            x=F.normalize(x, p=2, dim=1)
            # x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _,_ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _,_ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not False:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        # x = metric.transform(x)
        # y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m, x.numpy(), y.numpy()


def _stack_feature_tensor(features, entries):
    stacked = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in entries], 0)
    return stacked.view(stacked.size(0), -1).float().contiguous()


def _pairwise_distance_block(query_block, gallery_block):
    query_sq = torch.pow(query_block, 2).sum(dim=1, keepdim=True)
    gallery_sq = torch.pow(gallery_block, 2).sum(dim=1, keepdim=True).t()
    dist_block = query_sq + gallery_sq
    dist_block.addmm_(query_block, gallery_block.t(), beta=1, alpha=-2)
    return dist_block


def _evaluate_place_blockwise_from_features(features, query, gallery, metric=False,
                                            topk=(1, 5, 10), cmc_flag=False,
                                            query_block_size=128, gallery_block_size=4096):
    eval_begin = time.time()
    query_tensor = _stack_feature_tensor(features, query)
    gallery_tensor = _stack_feature_tensor(features, gallery)
    if metric is not False:
        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        gallery_tensor = F.normalize(gallery_tensor, p=2, dim=1)

    query_ids = np.asarray([pid for _, pid, _, _ in query])
    gallery_ids = np.asarray([pid for _, pid, _, _ in gallery])
    gallery_pid_to_indices = {}
    for gallery_index, gallery_pid in enumerate(gallery_ids):
        gallery_pid_to_indices.setdefault(int(gallery_pid), []).append(gallery_index)
    gallery_pid_to_indices = {
        pid: np.asarray(indices, dtype=np.int64)
        for pid, indices in gallery_pid_to_indices.items()
    }

    max_topk = int(max(topk))
    num_valid_queries = 0
    ap_scores = []
    recall_hits = {int(k): 0.0 for k in topk}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gallery_device = None
    try:
        gallery_device = gallery_tensor.to(device, non_blocking=True)
    except RuntimeError:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        device = torch.device('cpu')
        gallery_device = gallery_tensor.to(device)

    num_gallery = gallery_tensor.size(0)

    for query_start in range(0, query_tensor.size(0), query_block_size):
        query_end = min(query_start + query_block_size, query_tensor.size(0))
        query_block = query_tensor[query_start:query_end].to(device, non_blocking=True)
        query_ids_block = query_ids[query_start:query_end]
        block_size = query_end - query_start

        positive_indices_block = []
        positive_counts = np.zeros((block_size,), dtype=np.int64)
        positive_dist_lists = [[] for _ in range(block_size)]
        valid_queries_block = np.zeros((block_size,), dtype=bool)

        for local_index, query_pid in enumerate(query_ids_block):
            indices = gallery_pid_to_indices.get(int(query_pid))
            if indices is None or len(indices) == 0:
                positive_indices_block.append(np.empty((0,), dtype=np.int64))
                continue
            positive_indices_block.append(indices)
            positive_counts[local_index] = len(indices)
            valid_queries_block[local_index] = True

        topk_vals = None
        topk_indices = None

        for gallery_start in range(0, num_gallery, gallery_block_size):
            gallery_end = min(gallery_start + gallery_block_size, num_gallery)
            gallery_block = gallery_device[gallery_start:gallery_end]
            dist_block = _pairwise_distance_block(query_block, gallery_block)

            block_k = min(max_topk, dist_block.size(1))
            block_topk_vals, block_topk_rel_indices = torch.topk(
                dist_block, k=block_k, dim=1, largest=False, sorted=True
            )
            block_topk_indices = block_topk_rel_indices + gallery_start

            if topk_vals is None:
                topk_vals = block_topk_vals
                topk_indices = block_topk_indices
            else:
                merged_vals = torch.cat((topk_vals, block_topk_vals), dim=1)
                merged_indices = torch.cat((topk_indices, block_topk_indices), dim=1)
                topk_vals, merged_order = torch.topk(
                    merged_vals, k=max_topk, dim=1, largest=False, sorted=True
                )
                topk_indices = merged_indices.gather(1, merged_order)

            for local_index, positive_indices in enumerate(positive_indices_block):
                if positive_indices.size == 0:
                    continue
                left = np.searchsorted(positive_indices, gallery_start, side='left')
                right = np.searchsorted(positive_indices, gallery_end, side='left')
                if left >= right:
                    continue
                rel_indices = positive_indices[left:right] - gallery_start
                positive_dist_lists[local_index].extend(
                    dist_block[local_index, rel_indices].detach().cpu().tolist()
                )

        if topk_indices is None:
            continue

        sorted_positive_dists = []
        max_positive_count = 0
        for local_index, positive_values in enumerate(positive_dist_lists):
            if not valid_queries_block[local_index]:
                sorted_positive_dists.append([])
                continue
            sorted_values = sorted(positive_values)
            sorted_positive_dists.append(sorted_values)
            max_positive_count = max(max_positive_count, len(sorted_values))

        less_counts_cpu = None
        if max_positive_count > 0:
            threshold_cpu = np.full((block_size, max_positive_count), np.inf, dtype=np.float32)
            valid_mask_cpu = np.zeros((block_size, max_positive_count), dtype=np.bool_)
            for local_index, sorted_values in enumerate(sorted_positive_dists):
                if not sorted_values:
                    continue
                count = len(sorted_values)
                threshold_cpu[local_index, :count] = np.asarray(sorted_values, dtype=np.float32)
                valid_mask_cpu[local_index, :count] = True

            threshold_tensor = torch.from_numpy(threshold_cpu).to(device, non_blocking=True)
            valid_mask_tensor = torch.from_numpy(valid_mask_cpu).to(device, non_blocking=True)
            less_counts = torch.zeros((block_size, max_positive_count), dtype=torch.int64, device=device)

            for gallery_start in range(0, num_gallery, gallery_block_size):
                gallery_end = min(gallery_start + gallery_block_size, num_gallery)
                gallery_block = gallery_device[gallery_start:gallery_end]
                dist_block = _pairwise_distance_block(query_block, gallery_block)
                compare = dist_block.unsqueeze(2) < threshold_tensor.unsqueeze(1)
                compare = compare & valid_mask_tensor.unsqueeze(1)
                less_counts += compare.sum(dim=1)

            less_counts_cpu = less_counts.detach().cpu().numpy()

        topk_indices_cpu = topk_indices.detach().cpu().numpy()
        for local_index, is_valid in enumerate(valid_queries_block):
            if not is_valid:
                continue
            num_valid_queries += 1

            query_pid = int(query_ids_block[local_index])
            retrieved_gallery_ids = gallery_ids[topk_indices_cpu[local_index]]
            matches = (retrieved_gallery_ids == query_pid)
            for k in topk:
                recall_hits[int(k)] += float(matches[:int(k)].any())

            positive_count = int(positive_counts[local_index])
            if positive_count <= 0 or less_counts_cpu is None:
                continue
            positive_less = less_counts_cpu[local_index, :positive_count].astype(np.float32)
            positive_order = np.arange(1, positive_count + 1, dtype=np.float32)
            ap_scores.append(float(np.mean(positive_order / (positive_less + positive_order))))

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if num_valid_queries == 0:
        raise RuntimeError('No valid place query with a matching gallery position')

    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    recalls = {int(k): recall_hits[int(k)] / float(num_valid_queries) for k in topk}

    print('Place mAP: {:4.1%}'.format(mAP))
    if not cmc_flag:
        print('Place blockwise evaluation time: {:.2f}s'.format(time.time() - eval_begin))
        return mAP
    print('Recall@K:')
    for k in topk:
        print('  R@{:<4}{:12.1%}'.format(k, recalls[int(k)]))
    print('Place blockwise evaluation time: {:.2f}s'.format(time.time() - eval_begin))
    return recalls[int(topk[0])], mAP


def _evaluate_place(distmat, query_ids, gallery_ids, topk=(1, 5, 10), cmc_flag=False):
    distmat = _to_numpy_distmat(distmat)
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)

    if distmat.ndim != 2:
        raise ValueError('Place distance matrix must be 2D, got {}'.format(distmat.ndim))

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    valid_queries = matches.any(axis=1)
    num_valid_queries = int(valid_queries.sum())
    if num_valid_queries == 0:
        raise RuntimeError('No valid place query with a matching gallery position')

    ap_scores = []
    recall_scores = {int(k): 0.0 for k in topk}
    for row_matches, is_valid in zip(matches, valid_queries):
        if not is_valid:
            continue
        positive_ranks = np.flatnonzero(row_matches)
        precisions = [(rank_idx + 1.0) / float(rank + 1.0) for rank_idx, rank in enumerate(positive_ranks)]
        ap_scores.append(float(np.mean(precisions)))
        for k in recall_scores:
            recall_scores[k] += float(row_matches[:k].any())

    mAP = float(np.mean(ap_scores))
    recalls = {k: recall_scores[k] / float(num_valid_queries) for k in recall_scores}

    print('Place mAP: {:4.1%}'.format(mAP))
    if not cmc_flag:
        return mAP
    print('Recall@K:')
    for k in topk:
        print('  R@{:<4}{:12.1%}'.format(k, recalls[int(k)]))
    return recalls[int(topk[0])], mAP


def _extract_place_local_ids(samples):
    local_ids = []
    for fpath, _, _, _ in samples:
        stem = osp.splitext(osp.basename(str(fpath)))[0]
        matches = re.findall(r"\d+", stem)
        if matches:
            local_ids.append(int(matches[-1]))
        else:
            local_ids.append(None)
    return local_ids


def _evaluate_place_with_neighbor_tolerance(
        distmat, query, gallery, dataset_name, topk=(1, 5, 10), cmc_flag=False):
    if dataset_name != 'robotcar_place':
        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        return _evaluate_place(distmat, query_ids, gallery_ids, topk=topk, cmc_flag=cmc_flag)

    distmat = _to_numpy_distmat(distmat)
    query_local_ids = _extract_place_local_ids(query)
    gallery_local_ids = _extract_place_local_ids(gallery)
    query_ids = np.asarray([pid for _, pid, _, _ in query])
    gallery_ids = np.asarray([pid for _, pid, _, _ in gallery])

    gallery_pid_to_indices = {}
    for gallery_index, gallery_pid in enumerate(gallery_ids):
        gallery_pid_to_indices.setdefault(int(gallery_pid), []).append(gallery_index)
    gallery_pid_to_indices = {
        pid: np.asarray(indices, dtype=np.int64)
        for pid, indices in gallery_pid_to_indices.items()
    }

    gallery_local_to_indices = {}
    for gallery_index, local_id in enumerate(gallery_local_ids):
        if local_id is None:
            continue
        gallery_local_to_indices.setdefault(int(local_id), []).append(gallery_index)
    gallery_local_to_indices = {
        local_id: np.asarray(indices, dtype=np.int64)
        for local_id, indices in gallery_local_to_indices.items()
    }

    indices = np.argsort(distmat, axis=1)
    ap_scores = []
    recall_scores = {int(k): 0.0 for k in topk}
    num_valid_queries = 0

    for row_idx, ranked_indices in enumerate(indices):
        positive_index_set = set()
        strict_positive_indices = gallery_pid_to_indices.get(int(query_ids[row_idx]))
        if strict_positive_indices is not None:
            positive_index_set.update(strict_positive_indices.tolist())

        local_id = query_local_ids[row_idx]
        if local_id is not None:
            for delta in (-1, 0, 1):
                neighbor_indices = gallery_local_to_indices.get(int(local_id) + delta)
                if neighbor_indices is not None:
                    positive_index_set.update(neighbor_indices.tolist())

        if not positive_index_set:
            continue

        positive_index_set = set(int(x) for x in positive_index_set)
        row_matches = np.asarray([int(idx) in positive_index_set for idx in ranked_indices], dtype=bool)
        if not row_matches.any():
            continue

        num_valid_queries += 1
        positive_ranks = np.flatnonzero(row_matches)
        precisions = [(rank_idx + 1.0) / float(rank + 1.0) for rank_idx, rank in enumerate(positive_ranks)]
        ap_scores.append(float(np.mean(precisions)))
        for k in recall_scores:
            recall_scores[k] += float(row_matches[:int(k)].any())

    if num_valid_queries == 0:
        raise RuntimeError('No valid place query with a matching gallery position')

    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    recalls = {k: recall_scores[k] / float(num_valid_queries) for k in recall_scores}

    print('Place mAP: {:4.1%}'.format(mAP))
    print('Using ±1 index neighbor tolerance for {}'.format(dataset_name))
    if not cmc_flag:
        return mAP
    print('Recall@K:')
    for k in topk:
        print('  R@{:<4}{:12.1%}'.format(k, recalls[int(k)]))
    return recalls[int(topk[0])], mAP


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, cuhk03=False, task_type='reid',
                 dataset_name=None):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _,_ in query]
        gallery_ids = [pid for _, pid, _,_ in gallery]
        query_cams = [cam for _, _, cam,_ in query]
        gallery_cams = [cam for _, _, cam,_ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None)
        if task_type != 'place':
            assert (query_cams is not None and gallery_cams is not None)

    if task_type == 'place':
        if query is not None and gallery is not None:
            return _evaluate_place_with_neighbor_tolerance(
                distmat, query, gallery, dataset_name=dataset_name,
                topk=cmc_topk, cmc_flag=cmc_flag
            )
        return _evaluate_place(distmat, query_ids, gallery_ids, topk=cmc_topk, cmc_flag=cmc_flag)

    # Compute mean AP
    if cuhk03:
        mAP = mean_ap_cuhk03(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    else:
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP
    '''
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
        'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    '''
    if cuhk03:
        cmc_configs = {
        'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False)
                }
        
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
        print('CUHK03 CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                .format(k,
                        cmc_scores['cuhk03'][k-1]))
        return cmc_scores['cuhk03'][0], mAP
    
    else:
        cmc_configs = {
            'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
                }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
        if task_type == 'place':
            print('Recall@K:')
        else:
            print('CMC Scores:')
        for k in cmc_topk:
            if task_type == 'place':
                print('  R@{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
            else:
                print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
        return cmc_scores['market1501'][0], mAP

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False,
                 rerank=False, pre_features=None, cuhk03=False,training_phase=None,
                 dataset_name=None, external_matrix_dir=None, external_matrix_mode='replace',
                 external_matrix_weight=0.5, external_matrix_kind='distance',
                 external_matrix_manifest=None, task_type='reid',
                 external_embedding_manifest=None, external_embedding_mode='replace',
                 external_embedding_weight=0.5, fusion_strategy='linear',
                 fusion_rrf_k=60, adaptive_alpha_temp=10.0,
                 adaptive_confidence_mode='gap',
                 adaptive_alpha_min=0.0,
                 adaptive_alpha_max=1.0,
                 adaptive_gap_weight=1.0,
                 adaptive_entropy_weight=1.0,
                 adaptive_maxsim_weight=1.0,
                 adaptive_entropy_softmax_temp=10.0,
                 place_eval_mode='legacy'):
        manifest_item = _load_external_matrix_manifest(external_matrix_manifest, dataset_name)
        embedding_item = _load_external_embedding_manifest(external_embedding_manifest, dataset_name)
        if manifest_item is not None:
            qg_external_path = manifest_item.get('qg')
            qq_external_path = manifest_item.get('qq')
            gg_external_path = manifest_item.get('gg')
            if manifest_item.get('kind') is not None:
                external_matrix_kind = manifest_item['kind']
        else:
            qg_external_path = _resolve_external_matrix_path(external_matrix_dir, dataset_name, 'qg')
            qq_external_path = _resolve_external_matrix_path(external_matrix_dir, dataset_name, 'qq')
            gg_external_path = _resolve_external_matrix_path(external_matrix_dir, dataset_name, 'gg')
        use_external = qg_external_path is not None

        features = None
        distmat = None
        query_features = None
        gallery_features = None
        embedding_dist = None
        model_dist = None
        model_query_features = None
        model_gallery_features = None
        fusion_details = None

        if task_type == 'place' and place_eval_mode == 'blockwise' and embedding_item is None and not use_external and not rerank:
            if pre_features is None:
                features, _ = extract_features(self.model, data_loader, training_phase=training_phase)
            else:
                features = pre_features
            return _evaluate_place_blockwise_from_features(
                features,
                query,
                gallery,
                metric=metric,
                topk=(1, 5, 10),
                cmc_flag=cmc_flag,
            )

        if embedding_item is not None:
            query_embeddings, gallery_embeddings = _load_external_embeddings(
                embedding_item['query_embeddings'],
                embedding_item['gallery_embeddings'],
                expected_query=len(query),
                expected_gallery=len(gallery),
            )
            embedding_dist = _pairwise_distance_from_embeddings(
                query_embeddings,
                gallery_embeddings,
                metric=embedding_item.get('metric', 'cosine'),
                normalized=embedding_item.get('normalized', False),
            )
            print('Using external embeddings for dataset {} from {}'.format(dataset_name, external_embedding_manifest))

        if embedding_item is not None and external_embedding_mode == 'replace':
            distmat = embedding_dist
            query_features = query_embeddings
            gallery_features = gallery_embeddings
            if use_external:
                external_qg = _load_external_matrix(
                    qg_external_path,
                    expected_shape=(len(query), len(gallery)),
                    matrix_kind=external_matrix_kind
                )
                if external_matrix_mode == 'replace':
                    distmat = external_qg
                    print('Replacing external embedding distance matrix with external matrix for dataset {}'.format(dataset_name))
                elif external_matrix_mode == 'fuse':
                    distmat, fusion_details = _fuse_distance_matrices(
                        distmat, external_qg, weight=external_matrix_weight,
                        strategy=fusion_strategy, rrf_k=fusion_rrf_k,
                        adaptive_temp=adaptive_alpha_temp,
                        adaptive_confidence_mode=adaptive_confidence_mode,
                        adaptive_alpha_min=adaptive_alpha_min,
                        adaptive_alpha_max=adaptive_alpha_max,
                        adaptive_gap_weight=adaptive_gap_weight,
                        adaptive_entropy_weight=adaptive_entropy_weight,
                        adaptive_maxsim_weight=adaptive_maxsim_weight,
                        adaptive_entropy_softmax_temp=adaptive_entropy_softmax_temp,
                        return_details=True
                    )
                    print('Fusing external embedding distance matrix with external matrix for dataset {} (weight={:.2f}, strategy={})'.format(
                        dataset_name, external_matrix_weight, fusion_strategy
                    ))
                else:
                    raise ValueError('Unsupported external_matrix_mode: {}'.format(external_matrix_mode))
        elif embedding_item is not None and external_embedding_mode == 'fuse':
            if (pre_features is None):
                features, _ = extract_features(self.model, data_loader,training_phase=training_phase)
            else:
                features = pre_features
            model_dist, model_query_features, model_gallery_features = pairwise_distance(features, query, gallery, metric=metric)
            distmat, fusion_details = _fuse_distance_matrices(
                model_dist, embedding_dist, weight=external_embedding_weight,
                strategy=fusion_strategy, rrf_k=fusion_rrf_k,
                adaptive_temp=adaptive_alpha_temp,
                adaptive_confidence_mode=adaptive_confidence_mode,
                adaptive_alpha_min=adaptive_alpha_min,
                adaptive_alpha_max=adaptive_alpha_max,
                adaptive_gap_weight=adaptive_gap_weight,
                adaptive_entropy_weight=adaptive_entropy_weight,
                adaptive_maxsim_weight=adaptive_maxsim_weight,
                adaptive_entropy_softmax_temp=adaptive_entropy_softmax_temp,
                return_details=True
            )
            query_features = model_query_features
            gallery_features = model_gallery_features
            print('Fusing model distance matrix with external embeddings for dataset {} (model_weight={:.2f}, strategy={})'.format(
                dataset_name, external_embedding_weight, fusion_strategy
            ))
            if use_external:
                external_qg = _load_external_matrix(
                    qg_external_path,
                    expected_shape=(len(query), len(gallery)),
                    matrix_kind=external_matrix_kind
                )
                if external_matrix_mode == 'replace':
                    distmat = external_qg
                    print('Replacing fused model/external-embedding distance matrix with external matrix for dataset {}'.format(dataset_name))
                elif external_matrix_mode == 'fuse':
                    distmat, fusion_details = _fuse_distance_matrices(
                        distmat, external_qg, weight=external_matrix_weight,
                        strategy=fusion_strategy, rrf_k=fusion_rrf_k,
                        adaptive_temp=adaptive_alpha_temp,
                        adaptive_confidence_mode=adaptive_confidence_mode,
                        adaptive_alpha_min=adaptive_alpha_min,
                        adaptive_alpha_max=adaptive_alpha_max,
                        adaptive_gap_weight=adaptive_gap_weight,
                        adaptive_entropy_weight=adaptive_entropy_weight,
                        adaptive_maxsim_weight=adaptive_maxsim_weight,
                        adaptive_entropy_softmax_temp=adaptive_entropy_softmax_temp,
                        return_details=True
                    )
                    print('Fusing model/external-embedding distance matrix with external matrix for dataset {} (weight={:.2f}, strategy={})'.format(
                        dataset_name, external_matrix_weight, fusion_strategy
                    ))
                else:
                    raise ValueError('Unsupported external_matrix_mode: {}'.format(external_matrix_mode))
        elif use_external and external_matrix_mode == 'replace' and not rerank:
            distmat = _load_external_matrix(
                qg_external_path,
                expected_shape=(len(query), len(gallery)),
                matrix_kind=external_matrix_kind
            )
            print('Using external matrix for dataset {} from {}'.format(dataset_name, qg_external_path))
        else:
            if (pre_features is None):
                features, _ = extract_features(self.model, data_loader,training_phase=training_phase)
            else:
                features = pre_features
            distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
            if use_external:
                external_qg = _load_external_matrix(
                    qg_external_path,
                    expected_shape=(len(query), len(gallery)),
                    matrix_kind=external_matrix_kind
                )
                if external_matrix_mode == 'replace':
                    distmat = external_qg
                    print('Replacing model distance matrix with external matrix for dataset {}'.format(dataset_name))
                elif external_matrix_mode == 'fuse':
                    distmat, fusion_details = _fuse_distance_matrices(
                        distmat, external_qg, weight=external_matrix_weight,
                        strategy=fusion_strategy, rrf_k=fusion_rrf_k,
                        adaptive_temp=adaptive_alpha_temp,
                        adaptive_confidence_mode=adaptive_confidence_mode,
                        adaptive_alpha_min=adaptive_alpha_min,
                        adaptive_alpha_max=adaptive_alpha_max,
                        adaptive_gap_weight=adaptive_gap_weight,
                        adaptive_entropy_weight=adaptive_entropy_weight,
                        adaptive_maxsim_weight=adaptive_maxsim_weight,
                        adaptive_entropy_softmax_temp=adaptive_entropy_softmax_temp,
                        return_details=True
                    )
                    print('Fusing model distance matrix with external matrix for dataset {} (weight={:.2f}, strategy={})'.format(
                        dataset_name, external_matrix_weight, fusion_strategy
                    ))
                else:
                    raise ValueError('Unsupported external_matrix_mode: {}'.format(external_matrix_mode))

        if fusion_details is not None and fusion_details.get('strategy') == 'confidence_adaptive':
            print('Adaptive alpha stats for {}: mode={} temp={:.2f} bounds=({:.2f},{:.2f}) | mean={:.3f} min={:.3f} max={:.3f} | conf_model={:.4f} conf_external={:.4f}'.format(
                dataset_name,
                fusion_details['adaptive_confidence_mode'],
                fusion_details['adaptive_temp'],
                fusion_details['adaptive_alpha_min_bound'],
                fusion_details['adaptive_alpha_max_bound'],
                fusion_details['alpha_mean'],
                fusion_details['alpha_min'],
                fusion_details['alpha_max'],
                fusion_details['conf_primary_mean'],
                fusion_details['conf_secondary_mean'],
            ))

        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                               cmc_flag=cmc_flag, cuhk03=cuhk03, task_type=task_type,
                               dataset_name=dataset_name)
        if (not rerank):
            return results

        print('Applying re-ranking ...')
        if use_external and qq_external_path is not None and gg_external_path is not None:
            distmat_qq = _load_external_matrix(
                qq_external_path,
                expected_shape=(len(query), len(query)),
                matrix_kind=external_matrix_kind
            )
            distmat_gg = _load_external_matrix(
                gg_external_path,
                expected_shape=(len(gallery), len(gallery)),
                matrix_kind=external_matrix_kind
            )
        else:
            distmat_qq = pairwise_distance(features, query, query, metric=metric)
            distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)
            distmat_qq = _to_numpy_distmat(distmat_qq)
            distmat_gg = _to_numpy_distmat(distmat_gg)
        distmat = re_ranking(_to_numpy_distmat(distmat), distmat_qq, distmat_gg)
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                            cmc_flag=cmc_flag, task_type=task_type, dataset_name=dataset_name)

