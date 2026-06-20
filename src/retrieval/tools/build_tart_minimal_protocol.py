from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sorted_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def load_pose_xy(pose_path: Path):
    poses = []
    with pose_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            poses.append(np.asarray([float(parts[0]), float(parts[1])], dtype=np.float32))
    return poses


def scene_names(tart_root: Path):
    names = set()
    for child in tart_root.iterdir():
        if child.is_dir() and child.name.endswith("_Easy_image_left"):
            names.add(child.name[: -len("_Easy_image_left")])
    return sorted(names)


def easy_routes(seq_dir: Path):
    routes = []
    for pose_path in sorted(seq_dir.rglob("pose_left.txt")):
        route_dir = pose_path.parent
        image_dir = route_dir / "image_left"
        if not image_dir.exists():
            continue
        image_paths = sorted_images(image_dir)
        poses = load_pose_xy(pose_path)
        if len(image_paths) != len(poses):
            continue
        routes.append((route_dir, image_paths, poses))
    return routes


def merged_gallery_sequence(seq_dir: Path):
    routes = easy_routes(seq_dir)
    if not routes:
        raise RuntimeError(f"No valid Easy routes found for {seq_dir}")

    gallery_paths = []
    gallery_poses = []
    route_dirs = []
    for route_dir, route_images, route_poses in routes:
        gallery_paths.extend(route_images)
        gallery_poses.extend(route_poses)
        route_dirs.append(str(route_dir).replace("\\", "/"))
    return gallery_paths, gallery_poses, route_dirs


def iter_hard_routes(seq_dir: Path):
    for pose_path in sorted(seq_dir.rglob("pose_left.txt")):
        route_dir = pose_path.parent
        image_dir = route_dir / "image_left"
        if not image_dir.exists():
            continue
        image_paths = sorted_images(image_dir)
        poses = load_pose_xy(pose_path)
        if len(image_paths) != len(poses):
            continue
        yield route_dir, image_paths, poses


def build_scene_protocol(
    tart_root: Path,
    scene_name: str,
    topk_positive_radius: int = 0,
    positive_distance_threshold: float | None = None,
    fallback_nearest: bool = False,
):
    easy_dir = tart_root / f"{scene_name}_Easy_image_left"
    hard_dir = tart_root / f"{scene_name}_Hard_image_left"
    if not easy_dir.exists() or not hard_dir.exists():
        raise FileNotFoundError(f"Missing Easy/Hard dirs for scene {scene_name}")

    gallery_paths, gallery_pose, gallery_pose_routes = merged_gallery_sequence(easy_dir)
    gallery_pose_matrix = np.stack(gallery_pose, axis=0)

    query_paths = []
    gt_rows = []
    query_meta = []
    for route_dir, image_paths, poses in iter_hard_routes(hard_dir):
        route_name = route_dir.name
        for local_index, (img_path, query_xy) in enumerate(zip(image_paths, poses)):
            dist = np.linalg.norm(gallery_pose_matrix - query_xy[None, :], axis=1)
            best_index = int(np.argmin(dist))
            gt_row = np.zeros((len(gallery_paths),), dtype=np.uint8)
            if positive_distance_threshold is not None:
                gt_row[dist <= float(positive_distance_threshold)] = 1
                if fallback_nearest and not gt_row.any():
                    gt_row[best_index] = 1
            else:
                start = max(0, best_index - topk_positive_radius)
                end = min(len(gallery_paths), best_index + topk_positive_radius + 1)
                gt_row[start:end] = 1
            query_paths.append(img_path)
            gt_rows.append(gt_row)
            query_meta.append(
                {
                    "scene": scene_name,
                    "route_name": route_name,
                    "local_index": local_index,
                    "matched_gallery_index": best_index,
                    "matched_gallery_distance_xy": float(dist[best_index]),
                    "positive_count": int(gt_row.sum()),
                }
            )

    if not query_paths:
        raise RuntimeError(f"No hard query frames found for scene {scene_name}")

    gt = np.stack(gt_rows, axis=0)
    return {
        "scene": scene_name,
        "gallery_paths": gallery_paths,
        "query_paths": query_paths,
        "gt": gt,
        "positive_distance_threshold": positive_distance_threshold,
        "gallery_meta": {
            "pose_route_dirs": gallery_pose_routes,
            "gallery_count": len(gallery_paths),
        },
        "query_meta": query_meta,
    }


def save_protocol(output_dir: Path, protocol):
    scene_name = protocol["scene"]
    output_dir.mkdir(parents=True, exist_ok=True)

    query_list_path = output_dir / f"{scene_name}_query_paths.json"
    gallery_list_path = output_dir / f"{scene_name}_gallery_paths.json"
    gt_path = output_dir / f"{scene_name}_gt.npy"
    meta_path = output_dir / f"{scene_name}_protocol_summary.json"

    query_paths = [str(p.resolve()).replace("\\", "/") for p in protocol["query_paths"]]
    gallery_paths = [str(p.resolve()).replace("\\", "/") for p in protocol["gallery_paths"]]

    query_list_path.write_text(json.dumps(query_paths, indent=2), encoding="utf-8")
    gallery_list_path.write_text(json.dumps(gallery_paths, indent=2), encoding="utf-8")
    np.save(gt_path, protocol["gt"].astype(np.uint8))

    dists = [item["matched_gallery_distance_xy"] for item in protocol["query_meta"]]
    summary = OrderedDict(
        scene=scene_name,
        query_count=len(query_paths),
        gallery_count=len(gallery_paths),
        gt_shape=[int(protocol["gt"].shape[0]), int(protocol["gt"].shape[1])],
        gt_positive_radius=int((protocol["gt"][0].sum() - 1) // 2) if protocol["gt"].size else 0,
        gt_positive_distance_threshold=(
            None if protocol["positive_distance_threshold"] is None
            else float(protocol["positive_distance_threshold"])
        ),
        gt_positive_count_mean=float(np.mean([item["positive_count"] for item in protocol["query_meta"]])),
        gt_positive_count_median=float(np.median([item["positive_count"] for item in protocol["query_meta"]])),
        matched_distance_xy_mean=float(np.mean(dists)),
        matched_distance_xy_median=float(np.median(dists)),
        matched_distance_xy_p95=float(np.percentile(dists, 95)),
        query_paths=str(query_list_path.resolve()).replace("\\", "/"),
        gallery_paths=str(gallery_list_path.resolve()).replace("\\", "/"),
        gt_path=str(gt_path.resolve()).replace("\\", "/"),
        gallery_pose_route_dirs=protocol["gallery_meta"]["pose_route_dirs"],
    )
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build a minimal TART query/gallery/GT protocol.")
    parser.add_argument("--data-root", required=True, help="Dataset root that contains tart/")
    parser.add_argument("--scene", default="all", help="Scene name or all")
    parser.add_argument("--gt-positive-radius", type=int, default=0, help="Expand GT around nearest gallery index")
    parser.add_argument("--positive-distance-threshold", type=float, default=None,
                        help="Use all Easy frames within this xy distance as positives; falls back to nearest-neighbor radius when omitted")
    parser.add_argument("--fallback-nearest", action="store_true",
                        help="When using a distance threshold, force the nearest gallery frame to be positive if none are within threshold.")
    parser.add_argument("--output-dir", required=True, help="Where to write protocol files")
    return parser.parse_args()


def main():
    args = parse_args()
    tart_root = Path(args.data_root).resolve() / "tart"
    if not tart_root.exists():
        raise FileNotFoundError(f"TART root not found: {tart_root}")

    selected_scenes = scene_names(tart_root) if args.scene == "all" else [args.scene]
    summaries = OrderedDict()
    for scene_name in selected_scenes:
        protocol = build_scene_protocol(
            tart_root=tart_root,
            scene_name=scene_name,
            topk_positive_radius=args.gt_positive_radius,
            positive_distance_threshold=args.positive_distance_threshold,
            fallback_nearest=args.fallback_nearest,
        )
        summaries[scene_name] = save_protocol(Path(args.output_dir), protocol)

    manifest_path = Path(args.output_dir) / "tart_minimal_protocol_manifest.json"
    manifest_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path.resolve()).replace("\\", "/"), "scenes": list(summaries.keys())}, indent=2))


if __name__ == "__main__":
    main()
