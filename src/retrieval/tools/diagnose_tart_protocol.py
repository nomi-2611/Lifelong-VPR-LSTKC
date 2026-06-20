from __future__ import absolute_import, print_function

import argparse
import json
import os
import os.path as osp
from collections import OrderedDict
from pathlib import Path

import numpy as np


VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _scene_names(tart_root):
    scenes = set()
    for child in Path(tart_root).iterdir():
        if child.is_dir() and child.name.endswith('_Easy_image_left'):
            scenes.add(child.name[:-len('_Easy_image_left')])
    return sorted(scenes)


def _sorted_images(image_dir):
    return sorted([
        path for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    ])


def _load_pose_xy(pose_path):
    poses = []
    with open(pose_path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            poses.append([float(parts[0]), float(parts[1])])
    return np.asarray(poses, dtype=np.float32)


def _iter_routes(seq_dir):
    for pose_path in sorted(Path(seq_dir).rglob('pose_left.txt')):
        route_dir = pose_path.parent
        image_dir = route_dir / 'image_left'
        if not image_dir.exists():
            continue
        images = _sorted_images(image_dir)
        poses = _load_pose_xy(pose_path)
        if len(images) != len(poses):
            yield {
                'route_dir': str(route_dir).replace('\\', '/'),
                'valid': False,
                'image_count': int(len(images)),
                'pose_count': int(len(poses)),
                'images': [],
                'poses': np.empty((0, 2), dtype=np.float32),
            }
            continue
        yield {
            'route_dir': str(route_dir).replace('\\', '/'),
            'valid': True,
            'image_count': int(len(images)),
            'pose_count': int(len(poses)),
            'images': images,
            'poses': poses,
        }


def _load_sequence(seq_dir):
    routes = list(_iter_routes(seq_dir))
    valid_routes = [route for route in routes if route['valid']]
    if not valid_routes:
        return routes, [], np.empty((0, 2), dtype=np.float32)
    paths = []
    poses = []
    for route in valid_routes:
        paths.extend(route['images'])
        poses.append(route['poses'])
    return routes, paths, np.concatenate(poses, axis=0).astype(np.float32, copy=False)


def _nearest_stats(query_poses, gallery_poses, chunk_size=512):
    if query_poses.size == 0 or gallery_poses.size == 0:
        return np.empty((0,), dtype=np.float32)
    gallery_sq = np.sum(np.square(gallery_poses), axis=1, keepdims=True).T
    nearest = []
    for start in range(0, query_poses.shape[0], chunk_size):
        end = min(start + chunk_size, query_poses.shape[0])
        block = query_poses[start:end]
        block_sq = np.sum(np.square(block), axis=1, keepdims=True)
        dist_sq = block_sq + gallery_sq - 2.0 * np.matmul(block, gallery_poses.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        nearest.extend(np.sqrt(np.min(dist_sq, axis=1)).astype(np.float32).tolist())
    return np.asarray(nearest, dtype=np.float32)


def _count_positives(query_poses, gallery_poses, thresholds, chunk_size=256):
    payload = {float(threshold): [] for threshold in thresholds}
    if query_poses.size == 0 or gallery_poses.size == 0:
        return {str(threshold): {'mean': 0.0, 'median': 0.0, 'zero_query_count': 0} for threshold in thresholds}
    gallery_sq = np.sum(np.square(gallery_poses), axis=1, keepdims=True).T
    for start in range(0, query_poses.shape[0], chunk_size):
        end = min(start + chunk_size, query_poses.shape[0])
        block = query_poses[start:end]
        block_sq = np.sum(np.square(block), axis=1, keepdims=True)
        dist_sq = block_sq + gallery_sq - 2.0 * np.matmul(block, gallery_poses.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        for threshold in thresholds:
            counts = np.sum(dist_sq <= float(threshold) * float(threshold), axis=1)
            payload[float(threshold)].extend(counts.astype(np.int64).tolist())
    summary = OrderedDict()
    for threshold in thresholds:
        counts = np.asarray(payload[float(threshold)], dtype=np.int64)
        summary[str(float(threshold))] = OrderedDict(
            mean=float(np.mean(counts)) if counts.size else 0.0,
            median=float(np.median(counts)) if counts.size else 0.0,
            min=int(np.min(counts)) if counts.size else 0,
            max=int(np.max(counts)) if counts.size else 0,
            zero_query_count=int(np.sum(counts == 0)) if counts.size else 0,
            valid_query_count=int(np.sum(counts > 0)) if counts.size else 0,
        )
    return summary


def _scene_summary(tart_root, scene, thresholds):
    easy_dir = Path(tart_root) / '{}_Easy_image_left'.format(scene)
    hard_dir = Path(tart_root) / '{}_Hard_image_left'.format(scene)
    easy_routes, gallery_paths, gallery_poses = _load_sequence(easy_dir)
    hard_routes, query_paths, query_poses = _load_sequence(hard_dir)
    nearest = _nearest_stats(query_poses, gallery_poses)
    positive_summary = _count_positives(query_poses, gallery_poses, thresholds)
    return OrderedDict(
        scene=scene,
        easy_dir=str(easy_dir).replace('\\', '/'),
        hard_dir=str(hard_dir).replace('\\', '/'),
        easy_route_count=len(easy_routes),
        hard_route_count=len(hard_routes),
        easy_valid_route_count=sum(1 for route in easy_routes if route['valid']),
        hard_valid_route_count=sum(1 for route in hard_routes if route['valid']),
        gallery_count=len(gallery_paths),
        query_count=len(query_paths),
        nearest_distance_xy=OrderedDict(
            mean=float(np.mean(nearest)) if nearest.size else 0.0,
            median=float(np.median(nearest)) if nearest.size else 0.0,
            p90=float(np.percentile(nearest, 90)) if nearest.size else 0.0,
            p95=float(np.percentile(nearest, 95)) if nearest.size else 0.0,
            max=float(np.max(nearest)) if nearest.size else 0.0,
        ),
        positives_by_threshold_xy=positive_summary,
        easy_routes=[
            OrderedDict(
                route_dir=route['route_dir'],
                valid=bool(route['valid']),
                image_count=int(route['image_count']),
                pose_count=int(route['pose_count']),
            )
            for route in easy_routes
        ],
        hard_routes=[
            OrderedDict(
                route_dir=route['route_dir'],
                valid=bool(route['valid']),
                image_count=int(route['image_count']),
                pose_count=int(route['pose_count']),
            )
            for route in hard_routes
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose TART Easy/Hard VPR protocol geometry.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--scene', default='all')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.25, 0.5, 1.0, 3.0, 5.0])
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tart_root = osp.join(args.data_root, 'tart')
    if not osp.isdir(tart_root):
        raise FileNotFoundError('TART root not found: {}'.format(tart_root))
    scenes = _scene_names(tart_root) if args.scene == 'all' else [args.scene]
    summary = OrderedDict()
    for scene in scenes:
        summary[scene] = _scene_summary(tart_root, scene, args.thresholds)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({
        'output_json': str(output_path.resolve()).replace('\\', '/'),
        'scenes': scenes,
    }, indent=2))


if __name__ == '__main__':
    main()
