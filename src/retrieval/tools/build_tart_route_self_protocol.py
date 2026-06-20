from __future__ import absolute_import, print_function

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _sorted_images(image_dir):
    return sorted([
        path for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    ])


def _load_routes(seq_dir):
    routes = []
    for pose_path in sorted(Path(seq_dir).rglob('pose_left.txt')):
        route_dir = pose_path.parent
        image_dir = route_dir / 'image_left'
        if not image_dir.exists():
            continue
        images = _sorted_images(image_dir)
        with open(pose_path, 'r', encoding='utf-8', errors='replace') as handle:
            poses = [line for line in handle if line.strip()]
        if len(images) == len(poses) and len(images) > 1:
            routes.append((route_dir, images))
    return routes


def parse_args():
    parser = argparse.ArgumentParser(description='Build a TART same-route self retrieval sanity protocol.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--scene', default='amusement')
    parser.add_argument('--difficulty', default='Easy', choices=['Easy', 'Hard'])
    parser.add_argument('--route-index', type=int, default=0)
    parser.add_argument('--query-stride', type=int, default=2)
    parser.add_argument('--gallery-stride', type=int, default=2)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seq_dir = Path(args.data_root) / 'tart' / '{}_{}_image_left'.format(args.scene, args.difficulty)
    routes = _load_routes(seq_dir)
    if not routes:
        raise RuntimeError('No valid routes found in {}'.format(seq_dir))
    route_index = min(max(0, int(args.route_index)), len(routes) - 1)
    route_dir, images = routes[route_index]

    query_indices = list(range(0, len(images), max(1, int(args.query_stride))))
    gallery_indices = list(range(0, len(images), max(1, int(args.gallery_stride))))
    gallery_index_to_position = {idx: pos for pos, idx in enumerate(gallery_indices)}

    query_paths = [str(images[idx].resolve()).replace('\\', '/') for idx in query_indices if idx in gallery_index_to_position]
    kept_query_indices = [idx for idx in query_indices if idx in gallery_index_to_position]
    gallery_paths = [str(images[idx].resolve()).replace('\\', '/') for idx in gallery_indices]
    gt = np.zeros((len(query_paths), len(gallery_paths)), dtype=np.uint8)
    for row, idx in enumerate(kept_query_indices):
        gt[row, gallery_index_to_position[idx]] = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / '{}_query_paths.json'.format(args.scene)).write_text(json.dumps(query_paths, indent=2), encoding='utf-8')
    (output_dir / '{}_gallery_paths.json'.format(args.scene)).write_text(json.dumps(gallery_paths, indent=2), encoding='utf-8')
    np.save(output_dir / '{}_gt.npy'.format(args.scene), gt)
    summary = OrderedDict(
        scene=args.scene,
        difficulty=args.difficulty,
        route_dir=str(route_dir).replace('\\', '/'),
        route_count=len(routes),
        route_index=route_index,
        query_count=len(query_paths),
        gallery_count=len(gallery_paths),
        gt_shape=[int(gt.shape[0]), int(gt.shape[1])],
    )
    (output_dir / '{}_protocol_summary.json'.format(args.scene)).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
