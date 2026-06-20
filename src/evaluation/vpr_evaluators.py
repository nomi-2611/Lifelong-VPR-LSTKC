from __future__ import absolute_import

import os
import os.path as osp
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.models.feature_extraction import extract_cnn_feature


_TART_POSE_CACHE = {}


def extract_features_for_vpr(model, data_loader, training_phase=None):
    phase_begin = time.time()
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    batch_count = 0

    with torch.no_grad():
        for imgs, fnames, pids, cids, domains in data_loader:
            batch_count += 1
            outputs = extract_cnn_feature(model, imgs, training_phase=training_phase)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output.float()
                labels[fname] = pid
    print('VPR eval phase [feature_extraction]: {:.2f}s, batches={}, features={}'.format(
        time.time() - phase_begin, int(batch_count), int(len(features))))
    return features, labels


def _stack_feature_tensor(features, entries):
    stacked = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in entries], 0)
    return stacked.view(stacked.size(0), -1).float().contiguous()


def _pairwise_distance(query_tensor, gallery_tensor, metric=False):
    if metric is not False:
        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        gallery_tensor = F.normalize(gallery_tensor, p=2, dim=1)
    query_sq = torch.pow(query_tensor, 2).sum(dim=1, keepdim=True)
    gallery_sq = torch.pow(gallery_tensor, 2).sum(dim=1, keepdim=True).t()
    distmat = query_sq + gallery_sq
    distmat.addmm_(query_tensor, gallery_tensor.t(), beta=1, alpha=-2)
    return distmat.cpu().numpy()


def _pairwise_distance_block(query_block, gallery_block):
    query_sq = torch.pow(query_block, 2).sum(dim=1, keepdim=True)
    gallery_sq = torch.pow(gallery_block, 2).sum(dim=1, keepdim=True).t()
    dist_block = query_sq + gallery_sq
    dist_block.addmm_(query_block, gallery_block.t(), beta=1, alpha=-2)
    return dist_block


def _parse_local_index(path):
    stem = osp.splitext(osp.basename(str(path)))[0]
    digits = ''.join(ch if ch.isdigit() else ' ' for ch in stem).split()
    if not digits:
        return None
    return int(digits[-1])


def _normalize_path(path):
    return osp.normcase(osp.normpath(str(path)))


def _load_pose_xy(pose_path):
    poses = []
    with open(pose_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            poses.append(np.asarray([float(parts[0]), float(parts[1])], dtype=np.float32))
    return poses


def _sorted_route_images(image_dir):
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        [
            str((Path(image_dir) / filename).resolve())
            for filename in os.listdir(image_dir)
            if (Path(filename).suffix.lower() in valid_exts)
        ]
    )


def _extract_tart_scene_key(path):
    parts = Path(str(path)).parts
    for part in parts:
        if part.endswith("_Easy_image_left") or part.endswith("_Hard_image_left"):
            return re.sub(r"_(Easy|Hard)_image_left$", "", part)
    return None


def _load_tart_route_pose_map(image_path):
    normalized = _normalize_path(image_path)
    image_dir = Path(image_path).resolve().parent
    if image_dir.name != "image_left":
        return {}
    route_dir = image_dir.parent
    cache_key = _normalize_path(route_dir)
    cached = _TART_POSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pose_path = route_dir / "pose_left.txt"
    if not pose_path.exists():
        _TART_POSE_CACHE[cache_key] = {}
        return _TART_POSE_CACHE[cache_key]

    image_paths = _sorted_route_images(image_dir)
    poses = _load_pose_xy(str(pose_path))
    if len(image_paths) != len(poses):
        _TART_POSE_CACHE[cache_key] = {}
        return _TART_POSE_CACHE[cache_key]

    payload = {
        _normalize_path(path): pose_xy
        for path, pose_xy in zip(image_paths, poses)
    }
    _TART_POSE_CACHE[cache_key] = payload
    return payload


def _extract_tart_pose_xy(image_path):
    route_map = _load_tart_route_pose_map(image_path)
    return route_map.get(_normalize_path(image_path))


def _build_pid_index(entries):
    pid_to_indices = {}
    for idx, (_, pid, _, _) in enumerate(entries):
        pid_to_indices.setdefault(int(pid), []).append(idx)
    return pid_to_indices


def _build_local_index(entries):
    local_to_indices = {}
    local_ids = []
    for idx, item in enumerate(entries):
        local_id = _parse_local_index(item[0])
        local_ids.append(local_id)
        if local_id is None:
            continue
        local_to_indices.setdefault(int(local_id), []).append(idx)
    return local_ids, local_to_indices


def _extract_at_metadata_xy(path):
    parts = osp.basename(str(path)).split('@')
    if len(parts) < 4:
        return None
    try:
        return np.asarray([float(parts[1]), float(parts[2])], dtype=np.float32)
    except ValueError:
        return None


def _resolve_protocol_settings(
        dataset_name,
        protocol,
        nordland_tolerance,
        robotcar_tolerance,
        tart_pose_threshold):
    if protocol not in {'dataset', 'strict_pid'}:
        raise ValueError('Unsupported VPR protocol: {}'.format(protocol))

    settings = {
        'use_pid_matches': True,
        'local_tolerance': 0,
        'protocol_name': protocol,
        'tart_pose_threshold': None,
        'pitts_pose_threshold': None,
    }
    if protocol == 'strict_pid':
        return settings

    settings['protocol_name'] = '{}_default'.format(dataset_name)
    if dataset_name == 'nordland_place':
        settings['local_tolerance'] = int(nordland_tolerance)
    elif dataset_name == 'robotcar_place':
        settings['local_tolerance'] = int(robotcar_tolerance)
    elif dataset_name == 'tart_place':
        settings['use_pid_matches'] = False
        settings['tart_pose_threshold'] = float(tart_pose_threshold)
        settings['protocol_name'] = 'tart_pose_radius'
    elif dataset_name == 'pitts30k_place':
        settings['use_pid_matches'] = False
        settings['pitts_pose_threshold'] = 25.0
        settings['protocol_name'] = 'pitts30k_utm_radius_25m'
    return settings


def _build_xy_radius_positive_map(query, gallery, radius):
    radius_sq = float(radius) * float(radius)
    positive_map = [np.empty((0,), dtype=np.int64) for _ in range(len(query))]
    gallery_items = []
    for gallery_index, item in enumerate(gallery):
        xy = _extract_at_metadata_xy(item[0])
        if xy is not None:
            gallery_items.append((int(gallery_index), xy))
    if not gallery_items:
        return positive_map

    gallery_indices = np.asarray([item[0] for item in gallery_items], dtype=np.int64)
    gallery_xy = np.stack([item[1] for item in gallery_items], axis=0).astype(np.float32, copy=False)
    gallery_sq = np.sum(np.square(gallery_xy), axis=1, keepdims=True).T

    query_chunk_size = 512
    query_items = []
    for query_index, item in enumerate(query):
        xy = _extract_at_metadata_xy(item[0])
        if xy is not None:
            query_items.append((int(query_index), xy))
    if not query_items:
        return positive_map

    query_indices = np.asarray([item[0] for item in query_items], dtype=np.int64)
    query_xy = np.stack([item[1] for item in query_items], axis=0).astype(np.float32, copy=False)
    for start in range(0, query_xy.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query_xy.shape[0])
        query_block = query_xy[start:end]
        query_index_block = query_indices[start:end]
        query_sq = np.sum(np.square(query_block), axis=1, keepdims=True)
        dist_sq = query_sq + gallery_sq - 2.0 * np.matmul(query_block, gallery_xy.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        within_radius = dist_sq <= radius_sq
        nearest_gallery = np.argmin(dist_sq, axis=1)
        for block_offset, query_index in enumerate(query_index_block):
            matched = np.flatnonzero(within_radius[block_offset])
            if matched.size > 0:
                positive_map[int(query_index)] = gallery_indices[matched]
            else:
                nearest = int(nearest_gallery[block_offset])
                positive_map[int(query_index)] = gallery_indices[nearest:nearest + 1]
    return positive_map


def _build_tart_positive_map(query, gallery, pose_threshold):
    pose_threshold_sq = float(pose_threshold) * float(pose_threshold)
    positive_map = [np.empty((0,), dtype=np.int64) for _ in range(len(query))]

    gallery_scene_to_indices = {}
    gallery_scene_to_poses = {}
    for gallery_index, item in enumerate(gallery):
        scene_key = _extract_tart_scene_key(item[0])
        pose_xy = _extract_tart_pose_xy(item[0])
        if scene_key is None or pose_xy is None:
            continue
        gallery_scene_to_indices.setdefault(scene_key, []).append(int(gallery_index))
        gallery_scene_to_poses.setdefault(scene_key, []).append(np.asarray(pose_xy, dtype=np.float32))

    query_scene_to_items = {}
    for query_index, item in enumerate(query):
        scene_key = _extract_tart_scene_key(item[0])
        pose_xy = _extract_tart_pose_xy(item[0])
        if scene_key is None or pose_xy is None:
            continue
        query_scene_to_items.setdefault(scene_key, []).append((int(query_index), np.asarray(pose_xy, dtype=np.float32)))

    query_chunk_size = 512
    for scene_key, query_items in query_scene_to_items.items():
        gallery_indices = gallery_scene_to_indices.get(scene_key)
        gallery_poses = gallery_scene_to_poses.get(scene_key)
        if not gallery_indices or not gallery_poses:
            continue

        gallery_indices = np.asarray(gallery_indices, dtype=np.int64)
        gallery_pose_array = np.stack(gallery_poses, axis=0).astype(np.float32, copy=False)
        gallery_pose_sq = np.sum(np.square(gallery_pose_array), axis=1, keepdims=True).T

        query_indices = np.asarray([item[0] for item in query_items], dtype=np.int64)
        query_pose_array = np.stack([item[1] for item in query_items], axis=0).astype(np.float32, copy=False)

        for start in range(0, query_pose_array.shape[0], query_chunk_size):
            end = min(start + query_chunk_size, query_pose_array.shape[0])
            query_pose_block = query_pose_array[start:end]
            query_index_block = query_indices[start:end]

            query_pose_sq = np.sum(np.square(query_pose_block), axis=1, keepdims=True)
            dist_sq = query_pose_sq + gallery_pose_sq - 2.0 * np.matmul(query_pose_block, gallery_pose_array.T)
            dist_sq = np.maximum(dist_sq, 0.0)

            within_threshold = dist_sq <= pose_threshold_sq
            nearest_gallery = np.argmin(dist_sq, axis=1)

            for block_offset, query_index in enumerate(query_index_block):
                matched = np.flatnonzero(within_threshold[block_offset])
                if matched.size > 0:
                    positive_map[int(query_index)] = gallery_indices[matched]
                else:
                    positive_map[int(query_index)] = gallery_indices[nearest_gallery[block_offset]:nearest_gallery[block_offset] + 1]

    return positive_map


def _build_positive_map(
        dataset_name,
        query,
        gallery,
        protocol,
        nordland_tolerance,
        robotcar_tolerance,
        tart_pose_threshold):
    query_local, _ = _build_local_index(query)
    _, gallery_local_to_indices = _build_local_index(gallery)
    gallery_pid_to_indices = _build_pid_index(gallery)
    settings = _resolve_protocol_settings(
        dataset_name=dataset_name,
        protocol=protocol,
        nordland_tolerance=nordland_tolerance,
        robotcar_tolerance=robotcar_tolerance,
        tart_pose_threshold=tart_pose_threshold,
    )

    if dataset_name == 'tart_place' and settings['tart_pose_threshold'] is not None:
        return _build_tart_positive_map(
            query,
            gallery,
            pose_threshold=settings['tart_pose_threshold'],
        ), settings

    if dataset_name == 'pitts30k_place' and settings['pitts_pose_threshold'] is not None:
        return _build_xy_radius_positive_map(
            query,
            gallery,
            radius=settings['pitts_pose_threshold'],
        ), settings

    positive_map = []
    for q_idx, (_, q_pid, _, _) in enumerate(query):
        positives = set()
        if settings['use_pid_matches']:
            positives.update(gallery_pid_to_indices.get(int(q_pid), []))

        local_tolerance = int(settings['local_tolerance'])
        if local_tolerance > 0:
            if dataset_name in {'nordland_place', 'robotcar_place'}:
                local_id = int(q_pid)
                local_to_indices = gallery_pid_to_indices
            else:
                local_id = query_local[q_idx]
                local_to_indices = gallery_local_to_indices
            if local_id is not None:
                for delta in range(-local_tolerance, local_tolerance + 1):
                    positives.update(local_to_indices.get(int(local_id) + delta, []))

        positive_map.append(np.asarray(sorted(positives), dtype=np.int64))
    return positive_map, settings


def _evaluate_vpr_blockwise(
        query_tensor,
        gallery_tensor,
        positive_map,
        topk=(1, 5, 10),
        query_block_size=64,
        gallery_block_size=4096):
    phase_begin = time.time()
    max_topk = int(max(topk))
    num_valid_queries = 0
    ap_scores = []
    recall_hits = {int(k): 0.0 for k in topk}
    distance_block_count = 0
    metric_block_count = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    move_begin = time.time()
    try:
        gallery_device = gallery_tensor.to(device, non_blocking=True)
    except RuntimeError:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        device = torch.device('cpu')
        gallery_device = gallery_tensor.to(device)
    print('VPR eval phase [gallery_to_device]: {:.2f}s, device={}, gallery_shape={}'.format(
        time.time() - move_begin, device.type, tuple(gallery_tensor.shape)))

    num_gallery = gallery_tensor.size(0)
    num_queries = query_tensor.size(0)

    for query_start in range(0, num_queries, query_block_size):
        query_end = min(query_start + query_block_size, num_queries)
        query_block = query_tensor[query_start:query_end].to(device, non_blocking=True)
        positive_indices_block = [
            np.asarray(positive_map[idx], dtype=np.int64)
            for idx in range(query_start, query_end)
        ]
        block_size = query_end - query_start
        valid_queries_block = np.asarray([indices.size > 0 for indices in positive_indices_block], dtype=bool)

        topk_vals = None
        topk_indices = None
        positive_dist_lists = [[] for _ in range(block_size)]

        for gallery_start in range(0, num_gallery, gallery_block_size):
            gallery_end = min(gallery_start + gallery_block_size, num_gallery)
            gallery_block = gallery_device[gallery_start:gallery_end]
            dist_block = _pairwise_distance_block(query_block, gallery_block)
            distance_block_count += 1

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
                metric_block_count += 1
                compare = dist_block.unsqueeze(2) < threshold_tensor.unsqueeze(1)
                compare = compare & valid_mask_tensor.unsqueeze(1)
                less_counts += compare.sum(dim=1)

            less_counts_cpu = less_counts.detach().cpu().numpy()

        topk_indices_cpu = topk_indices.detach().cpu().numpy()
        for local_index, positive_indices in enumerate(positive_indices_block):
            if positive_indices.size == 0:
                continue
            num_valid_queries += 1
            matches = np.isin(topk_indices_cpu[local_index], positive_indices, assume_unique=False)
            for k in recall_hits:
                recall_hits[int(k)] += float(matches[:int(k)].any())

            if less_counts_cpu is None:
                ap_scores.append(0.0)
                continue
            positive_count = int(positive_indices.size)
            positive_less = less_counts_cpu[local_index, :positive_count].astype(np.float32)
            positive_order = np.arange(1, positive_count + 1, dtype=np.float32)
            ap_scores.append(float(np.mean(positive_order / (positive_less + positive_order))))

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if num_valid_queries == 0:
        raise RuntimeError('No valid VPR queries with positives')

    recalls = {int(k): recall_hits[int(k)] / float(num_valid_queries) for k in topk}
    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    print('VPR eval phase [ranking_metrics]: {:.2f}s, query_blocks={}, distance_blocks={}, metric_blocks={}'.format(
        time.time() - phase_begin,
        int((num_queries + query_block_size - 1) // query_block_size),
        int(distance_block_count),
        int(metric_block_count)))
    return recalls, mAP, num_valid_queries


def _evaluate_vpr_blockwise_fullsort(
        query_tensor,
        gallery_tensor,
        positive_map,
        topk=(1, 5, 10),
        query_block_size=64):
    phase_begin = time.time()
    max_topk = int(max(topk))
    num_valid_queries = 0
    ap_scores = []
    recall_hits = {int(k): 0.0 for k in topk}
    distance_block_count = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    move_begin = time.time()
    try:
        gallery_device = gallery_tensor.to(device, non_blocking=True)
    except RuntimeError:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        device = torch.device('cpu')
        gallery_device = gallery_tensor.to(device)
    print('VPR eval phase [gallery_to_device]: {:.2f}s, device={}, gallery_shape={}'.format(
        time.time() - move_begin, device.type, tuple(gallery_tensor.shape)))

    num_gallery = gallery_tensor.size(0)
    num_queries = query_tensor.size(0)

    for query_start in range(0, num_queries, query_block_size):
        query_end = min(query_start + query_block_size, num_queries)
        query_block = query_tensor[query_start:query_end].to(device, non_blocking=True)
        dist_block = _pairwise_distance_block(query_block, gallery_device)
        distance_block_count += 1

        sorted_indices = torch.argsort(dist_block, dim=1, descending=False)
        topk_indices_cpu = sorted_indices[:, :max_topk].detach().cpu().numpy()
        sorted_indices_cpu = sorted_indices.detach().cpu().numpy()

        for local_index, query_index in enumerate(range(query_start, query_end)):
            positive_indices = np.asarray(positive_map[query_index], dtype=np.int64)
            if positive_indices.size == 0:
                continue
            num_valid_queries += 1

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

    if num_valid_queries == 0:
        raise RuntimeError('No valid VPR queries with positives')

    recalls = {int(k): recall_hits[int(k)] / float(num_valid_queries) for k in topk}
    mAP = float(np.mean(ap_scores)) if ap_scores else 0.0
    print('VPR eval phase [ranking_metrics]: {:.2f}s, query_blocks={}, distance_blocks={}, metric_blocks={}'.format(
        time.time() - phase_begin,
        int((num_queries + query_block_size - 1) // query_block_size),
        int(distance_block_count),
        0))
    return recalls, mAP, num_valid_queries


def evaluate_vpr_dataset(
        model,
        data_loader,
        query,
        gallery,
        dataset_name,
        metric=False,
        training_phase=None,
        nordland_tolerance=0,
        robotcar_tolerance=1,
        tart_pose_threshold=0.25,
        protocol='dataset',
        topk=(1, 5, 10)):
    eval_begin = time.time()
    if dataset_name not in {'nordland_place', 'robotcar_place', 'tart_place', 'msls_place', 'pitts30k_place'}:
        raise ValueError('Unsupported VPR dataset for the adapter: {}'.format(dataset_name))

    features, _ = extract_features_for_vpr(model, data_loader, training_phase=training_phase)
    prep_begin = time.time()
    query_tensor = _stack_feature_tensor(features, query)
    gallery_tensor = _stack_feature_tensor(features, gallery)
    if metric is not False:
        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        gallery_tensor = F.normalize(gallery_tensor, p=2, dim=1)
    print('VPR eval phase [tensor_prep]: {:.2f}s, query_shape={}, gallery_shape={}'.format(
        time.time() - prep_begin, tuple(query_tensor.shape), tuple(gallery_tensor.shape)))

    positive_begin = time.time()
    positive_map, protocol_settings = _build_positive_map(
        dataset_name,
        query,
        gallery,
        protocol=protocol,
        nordland_tolerance=nordland_tolerance,
        robotcar_tolerance=robotcar_tolerance,
        tart_pose_threshold=tart_pose_threshold,
    )
    print('VPR eval phase [positive_map]: {:.2f}s, valid_positive_queries={}'.format(
        time.time() - positive_begin,
        int(sum(1 for positives in positive_map if positives.size > 0))))
    recalls, mAP, valid_queries = _evaluate_vpr_blockwise_fullsort(
        query_tensor,
        gallery_tensor,
        positive_map,
        topk=topk,
    )

    print('VPR protocol results on {}'.format(dataset_name))
    print('Protocol: {}'.format(protocol_settings['protocol_name']))
    if dataset_name == 'nordland_place':
        print('Nordland frame tolerance: +/-{}'.format(int(nordland_tolerance)))
    if dataset_name == 'robotcar_place':
        print('RobotCar frame tolerance: +/-{}'.format(int(robotcar_tolerance)))
    if dataset_name == 'tart_place' and protocol_settings['tart_pose_threshold'] is not None:
        print('TART pose threshold (xy): {:.3f}'.format(float(protocol_settings['tart_pose_threshold'])))
    if dataset_name == 'pitts30k_place' and protocol_settings['pitts_pose_threshold'] is not None:
        print('Pittsburgh UTM threshold (xy): {:.3f}m'.format(float(protocol_settings['pitts_pose_threshold'])))
    print('Valid queries: {}/{}'.format(int(valid_queries), int(len(query))))
    print('VPR mAP: {:4.1%}'.format(mAP))
    print('Recall@K:')
    for k in topk:
        print('  R@{:<4}{:12.1%}'.format(int(k), recalls[int(k)]))
    print('VPR eval phase [total]: {:.2f}s'.format(time.time() - eval_begin))
    return recalls[int(topk[0])], mAP


def evaluate_vpr_embeddings(
        query_embeddings,
        gallery_embeddings,
        query,
        gallery,
        dataset_name,
        metric=False,
        nordland_tolerance=0,
        robotcar_tolerance=1,
        tart_pose_threshold=0.25,
        protocol='dataset',
        topk=(1, 5, 10)):
    eval_begin = time.time()
    if dataset_name not in {'nordland_place', 'robotcar_place', 'tart_place', 'msls_place', 'pitts30k_place'}:
        raise ValueError('Unsupported VPR dataset for the adapter: {}'.format(dataset_name))

    prep_begin = time.time()
    query_tensor = torch.as_tensor(np.asarray(query_embeddings, dtype=np.float32)).contiguous()
    gallery_tensor = torch.as_tensor(np.asarray(gallery_embeddings, dtype=np.float32)).contiguous()
    if query_tensor.ndim != 2 or gallery_tensor.ndim != 2:
        raise ValueError('Query/gallery embeddings must be 2D arrays')
    if query_tensor.size(0) != len(query):
        raise ValueError('Query embedding count {} does not match query size {}'.format(
            int(query_tensor.size(0)), int(len(query))))
    if gallery_tensor.size(0) != len(gallery):
        raise ValueError('Gallery embedding count {} does not match gallery size {}'.format(
            int(gallery_tensor.size(0)), int(len(gallery))))
    if query_tensor.size(1) != gallery_tensor.size(1):
        raise ValueError('Embedding dimensions do not match: {} vs {}'.format(
            int(query_tensor.size(1)), int(gallery_tensor.size(1))))
    if metric is not False:
        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        gallery_tensor = F.normalize(gallery_tensor, p=2, dim=1)
    print('VPR eval phase [tensor_prep]: {:.2f}s, query_shape={}, gallery_shape={}'.format(
        time.time() - prep_begin, tuple(query_tensor.shape), tuple(gallery_tensor.shape)))

    positive_begin = time.time()
    positive_map, protocol_settings = _build_positive_map(
        dataset_name,
        query,
        gallery,
        protocol=protocol,
        nordland_tolerance=nordland_tolerance,
        robotcar_tolerance=robotcar_tolerance,
        tart_pose_threshold=tart_pose_threshold,
    )
    print('VPR eval phase [positive_map]: {:.2f}s, valid_positive_queries={}'.format(
        time.time() - positive_begin,
        int(sum(1 for positives in positive_map if positives.size > 0))))
    recalls, mAP, valid_queries = _evaluate_vpr_blockwise_fullsort(
        query_tensor,
        gallery_tensor,
        positive_map,
        topk=topk,
    )

    print('VPR protocol results on {}'.format(dataset_name))
    print('Protocol: {}'.format(protocol_settings['protocol_name']))
    if dataset_name == 'nordland_place':
        print('Nordland frame tolerance: +/-{}'.format(int(nordland_tolerance)))
    if dataset_name == 'robotcar_place':
        print('RobotCar frame tolerance: +/-{}'.format(int(robotcar_tolerance)))
    if dataset_name == 'tart_place' and protocol_settings['tart_pose_threshold'] is not None:
        print('TART pose threshold (xy): {:.3f}'.format(float(protocol_settings['tart_pose_threshold'])))
    if dataset_name == 'pitts30k_place' and protocol_settings['pitts_pose_threshold'] is not None:
        print('Pittsburgh UTM threshold (xy): {:.3f}m'.format(float(protocol_settings['pitts_pose_threshold'])))
    print('Valid queries: {}/{}'.format(int(valid_queries), int(len(query))))
    print('VPR mAP: {:4.1%}'.format(mAP))
    print('Recall@K:')
    for k in topk:
        print('  R@{:<4}{:12.1%}'.format(int(k), recalls[int(k)]))
    print('VPR eval phase [total]: {:.2f}s'.format(time.time() - eval_begin))
    return recalls[int(topk[0])], mAP
