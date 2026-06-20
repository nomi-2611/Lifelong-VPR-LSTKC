from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
VPRTEMPO_ROOT = CODE_ROOT / "feature_extraction" / "vprtempo_snn"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(VPRTEMPO_ROOT) not in sys.path:
    sys.path.insert(0, str(VPRTEMPO_ROOT))

from vprtempo_tools.export_qg_matrix import build_models, infer_model_structure, model_logger
from vprtempo.src.dataset import ProcessImage


def progress_log(output_dir, message, **payload):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    if payload:
        details = " ".join(f"{key}={value}" for key, value in payload.items())
        line = f"{line} | {details}"
    print(line, flush=True)
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    status = {"timestamp": timestamp, "message": message, **payload}
    (output_dir / "latest_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def resolve_eval_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def pose2mat(pose):
    t = pose[:, 0:3, None]
    rot = R.from_quat(pose[:, 3:7]).as_matrix().astype(np.float32).transpose(0, 2, 1)
    t = -rot @ t
    return torch.cat([torch.from_numpy(rot), torch.from_numpy(t)], dim=2)


def normalize_pixel_coordinates(points, height, width):
    scale_x = 2.0 / max(float(width - 1), 1.0)
    scale_y = 2.0 / max(float(height - 1), 1.0)
    x = points[..., 0] * scale_x - 1.0
    y = points[..., 1] * scale_y - 1.0
    return torch.stack([x, y], dim=-1)


def denormalize_pixel_coordinates(points, height, width):
    scale_x = max(float(width - 1), 1.0) / 2.0
    scale_y = max(float(height - 1), 1.0) / 2.0
    x = (points[..., 0] + 1.0) * scale_x
    y = (points[..., 1] + 1.0) * scale_y
    return torch.stack([x, y], dim=-1)


def convert_points_to_homogeneous(points):
    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)


def convert_points_from_homogeneous(points):
    z = points[..., 2:3].clamp(min=1e-8)
    return points[..., :2] / z


def transform_points(transform, points):
    points_h = convert_points_to_homogeneous(points)
    out = torch.matmul(transform, points_h.transpose(1, 2)).transpose(1, 2)
    return out[..., :3]


def inverse_transform(extrinsics):
    rot = extrinsics[:, :3, :3]
    trans = extrinsics[:, :3, 3:]
    rot_t = rot.transpose(1, 2)
    inv_trans = -torch.matmul(rot_t, trans)
    return torch.cat([rot_t, inv_trans], dim=2)


def compose_projection(K, pose):
    batch = K.shape[0]
    proj = torch.zeros((batch, 3, 4), dtype=K.dtype, device=K.device)
    proj[:, :, :3] = torch.matmul(K, pose[:, :, :3])
    proj[:, :, 3:] = torch.matmul(K, pose[:, :, 3:])
    return proj


def coord_list_grid_sample(values, points, mode="bilinear"):
    dim = len(points.shape)
    points = points.view(values.size(0), 1, -1, 2) if dim == 3 else points
    output = F.grid_sample(values, points, mode, align_corners=True).permute(0, 2, 3, 1)
    return output.squeeze(1) if dim == 3 else output


class Projector:
    @staticmethod
    def pix2world(points, depth_map, poses, Ks):
        height, width = depth_map.shape[2:]
        points_px = denormalize_pixel_coordinates(points, height, width)
        depths = coord_list_grid_sample(depth_map, points)[:, :, 0]
        points_h = convert_points_to_homogeneous(points_px)
        K_inv = torch.inverse(Ks)
        points_cam = torch.matmul(K_inv, points_h.transpose(1, 2)).transpose(1, 2) * depths.unsqueeze(-1)
        inv_pose = inverse_transform(poses)
        return transform_points(inv_pose, points_cam)

    @staticmethod
    def world2pix(points_world, resolution, poses, Ks, depth_map=None, eps=1e-2):
        height, width = resolution
        proj = compose_projection(Ks, poses)
        points_h = convert_points_to_homogeneous(points_world)
        points_proj = torch.matmul(proj, points_h.transpose(1, 2)).transpose(1, 2)
        xy_px = convert_points_from_homogeneous(points_proj)
        depth = points_proj[..., 2]
        xy = normalize_pixel_coordinates(xy_px, height, width)

        if depth_map is not None:
            depth_dst = coord_list_grid_sample(depth_map, xy)[:, :, 0]
            invalid = (depth < 0) | (depth > depth_dst + eps) | ((xy.abs() > 1).any(dim=-1))
            xy[invalid] = torch.nan
        return xy, depth


def src_repeat(x, n_dst):
    batch, shape = x.shape[0], x.shape[1:]
    return x.unsqueeze(1).expand(batch, n_dst, *shape).reshape(batch * n_dst, *shape)


def dst_repeat(x, n_src):
    batch, shape = x.shape[0], x.shape[1:]
    return x.unsqueeze(0).expand(n_src, batch, *shape).reshape(n_src * batch, *shape)


def feature_pt_ncovis(pos0, pts1, depth1, pose1, K1, eps=1e-2, grid_size=(12, 16)):
    b0, b1 = len(pos0), len(pts1)
    _, _, height, width = depth1.shape

    pts0_scr1, pts0_depth1 = Projector.world2pix(
        src_repeat(pos0, b1),
        (height, width),
        dst_repeat(pose1, b0),
        dst_repeat(K1, b0),
    )
    _, n_points, _ = pts0_scr1.shape
    pts0_scr1_depth1 = coord_list_grid_sample(
        depth1,
        pts0_scr1.reshape(b0, b1, n_points, 2).transpose(0, 1).reshape(b1, b0 * n_points, 2),
    ).reshape(b1, b0, n_points).transpose(0, 1).reshape(b0 * b1, n_points)
    invalid = (pts0_depth1 < 0) | (pts0_depth1 > pts0_scr1_depth1 + eps) | ((pts0_scr1.abs() > 1).any(dim=-1))
    pts0_scr1[invalid] = torch.nan

    ax = pts0_scr1.isfinite().all(dim=-1).to(torch.float32).mean(dim=-1)
    binned = denormalize_pixel_coordinates(pts0_scr1, *grid_size).round()
    valid_b, valid_n = binned.isfinite().all(dim=-1).nonzero(as_tuple=True)
    bp = torch.cat([valid_b[:, None], binned[valid_b, valid_n]], dim=-1) if valid_b.numel() else torch.empty((0, 3), device=binned.device)

    a1 = torch.zeros(b0 * b1, dtype=ax.dtype, device=ax.device)
    if bp.numel():
        batches, counts = bp.unique(dim=0)[:, 0].unique_consecutive(return_counts=True)
        a1[batches.to(torch.long)] = counts.to(a1.dtype) / float(np.prod(grid_size))

    covis = ax / (1.0 + ax / a1.clamp(min=1e-6) - ax)
    return covis.reshape(b0, b1)


def gen_probe(depth_map, scale=8):
    batch, _, height, width = depth_map.shape
    h = height // scale
    w = width // scale
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    points = torch.stack([xs + 0.5, ys + 0.5], dim=-1)
    points = normalize_pixel_coordinates(points, h + 1, w + 1).unsqueeze(0).expand(batch, -1, -1, -1)
    return points.reshape(batch, -1, 2).to(depth_map.device)


class TartanAirSequenceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.K = torch.tensor([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        self.ned2edn = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        self._load()

    def _load(self):
        for mode_dir in sorted(self.root.iterdir()):
            if not mode_dir.is_dir():
                continue
            for traj_dir in sorted(mode_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                pose_path = traj_dir / "pose_left.txt"
                image_dir = traj_dir / "image_left"
                depth_dir = traj_dir / "depth_left"
                if not pose_path.exists() or not image_dir.exists() or not depth_dir.exists():
                    continue
                pose_q = np.loadtxt(str(pose_path), dtype=np.float32)
                poses = torch.matmul(self.ned2edn, pose2mat(pose_q))
                image_names = sorted([p.name for p in image_dir.iterdir() if p.suffix.lower() == ".png"])
                depth_names = sorted([p.name for p in depth_dir.iterdir() if p.suffix.lower() == ".npy"])
                count = min(len(image_names), len(depth_names), poses.shape[0])
                seq_key = f"{mode_dir.name}_{traj_dir.name}"
                for idx in range(count):
                    self.samples.append(
                        {
                            "image_path": str((image_dir / image_names[idx]).resolve()),
                            "depth_path": str((depth_dir / depth_names[idx]).resolve()),
                            "pose": poses[idx],
                            "K": self.K.clone(),
                            "seq": seq_key,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = read_image(sample["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        depth = np.load(sample["depth_path"]).astype(np.float32)
        depth = torch.from_numpy(depth).unsqueeze(0)
        return image, depth, sample["pose"], sample["K"], sample["seq"]


def pairwise_cosine_chunked(embeddings, chunk_size=128):
    outputs = []
    for start0 in range(0, embeddings.shape[0], chunk_size):
        end0 = min(start0 + chunk_size, embeddings.shape[0])
        row = []
        x = embeddings[start0:end0]
        for start1 in range(0, embeddings.shape[0], chunk_size):
            end1 = min(start1 + chunk_size, embeddings.shape[0])
            y = embeddings[start1:end1]
            row.append(torch.matmul(x, y.t()).cpu())
        outputs.append(torch.cat(row, dim=1))
    return torch.cat(outputs, dim=0)


def build_embeddings_and_metadata(args):
    progress_log(args.output_dir, "embedding_stage_start", scene_root=args.scene_root)
    structure = infer_model_structure(args.model_path)
    logger, output_folder = model_logger()
    runtime_args = SimpleNamespace(
        dataset="viper_aligned_tartanair",
        data_dir=".",
        database_places=structure["database_places"],
        query_places=1,
        max_module=max(structure["out_dims"]),
        database_dirs="aligned_gallery",
        query_dir="aligned_query",
        GT_tolerance=0,
        skip=0,
        filter=1,
        epoch=1,
        patches=args.patches,
        dims=",".join(str(x) for x in args.dims),
        train_new_model=False,
        quantize=False,
        PR_curve=False,
        sim_mat=False,
        run_demo=False,
        export_matrix=False,
        export_prefix="viper_aligned_eval",
    )
    models = build_models(runtime_args, args.dims, logger, output_folder, structure)
    models[0].load_model(models, args.model_path)
    models[0].inferences = []
    for sub_model in models:
        models[0].inferences.append(
            torch.nn.Sequential(
                sub_model.feature_layer.w,
                sub_model.output_layer.w,
            ).to(torch.device(models[0].device))
        )
    image_transform = ProcessImage(args.dims, args.patches)

    dataset = TartanAirSequenceDataset(args.scene_root, transform=image_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    progress_log(args.output_dir, "dataset_loaded", samples=len(dataset))

    embeddings = []
    probe_points = []
    depths_all = []
    poses_all = []
    intrinsics_all = []
    seq_names = []

    with torch.no_grad():
        for sample_idx, (image, depth, pose, K, seq) in enumerate(loader, start=1):
            image = image.to(torch.device(models[0].device))
            out = models[0].forward(image)
            out = F.normalize(out.detach().cpu().squeeze(0).float(), dim=0)
            embeddings.append(out)

            depth = depth.float()
            pose = pose.float()
            K = K.float()
            probe = Projector.pix2world(gen_probe(depth), depth, pose, K).squeeze(0).cpu()
            probe_points.append(probe)
            depths_all.append(depth.squeeze(0).cpu())
            poses_all.append(pose.squeeze(0).cpu())
            intrinsics_all.append(K.squeeze(0).cpu())
            seq_names.append(seq[0])
            if sample_idx == 1 or sample_idx % args.log_every_samples == 0 or sample_idx == len(dataset):
                progress_log(
                    args.output_dir,
                    "embedding_progress",
                    done=sample_idx,
                    total=len(dataset),
                    pct=round(sample_idx * 100.0 / max(len(dataset), 1), 2),
                )

    embeddings = torch.stack(embeddings, dim=0)
    depths_all = torch.stack(depths_all, dim=0)
    poses_all = torch.stack(poses_all, dim=0)
    intrinsics_all = torch.stack(intrinsics_all, dim=0)
    progress_log(
        args.output_dir,
        "embedding_stage_done",
        embedding_shape=list(embeddings.shape),
        sequence_count=len(sorted(set(seq_names))),
    )
    return embeddings, probe_points, depths_all, poses_all, intrinsics_all, seq_names


def evaluate_streaming_r100p(
    output_dir,
    embeddings,
    probe_points,
    depths_all,
    poses_all,
    intrinsics_all,
    threshold,
    sim_chunk_size,
    gt_chunk_size,
    device,
    log_every_gallery_chunks,
):
    total = embeddings.shape[0]
    scores = []
    progress_log(
        output_dir,
        "stream_eval_start",
        total=total,
        gt_chunk_size=gt_chunk_size,
        sim_chunk_size=sim_chunk_size,
        device=str(device),
    )
    for q_start in range(0, total, gt_chunk_size):
        q_end = min(q_start + gt_chunk_size, total)
        query_embeddings = embeddings[q_start:q_end].to(device, non_blocking=True)
        query_probes = torch.stack(probe_points[q_start:q_end], dim=0).to(device, non_blocking=True)
        progress_log(
            output_dir,
            "query_chunk_start",
            q_start=q_start,
            q_end=q_end,
            total=total,
        )

        sim_parts = []
        rel_parts = []
        for g_start in range(0, total, sim_chunk_size):
            g_end = min(g_start + sim_chunk_size, total)
            gallery_embeddings = embeddings[g_start:g_end].to(device, non_blocking=True)
            sim_parts.append(torch.matmul(query_embeddings, gallery_embeddings.t()).cpu())

            adj = feature_pt_ncovis(
                query_probes,
                torch.zeros((g_end - g_start, 0, 3), dtype=query_probes.dtype, device=device),
                depths_all[g_start:g_end].to(device, non_blocking=True),
                poses_all[g_start:g_end].to(device, non_blocking=True),
                intrinsics_all[g_start:g_end].to(device, non_blocking=True),
            )
            rel_parts.append((adj.cpu() > threshold))
            if g_start == 0 or (g_start // sim_chunk_size) % log_every_gallery_chunks == 0 or g_end == total:
                progress_log(
                    output_dir,
                    "gallery_chunk_progress",
                    q_start=q_start,
                    q_end=q_end,
                    g_start=g_start,
                    g_end=g_end,
                    total=total,
                )

        sim_rows = torch.cat(sim_parts, dim=1)
        rel_rows = torch.cat(rel_parts, dim=1)

        for local_idx in range(sim_rows.shape[0]):
            global_idx = q_start + local_idx
            sim_row = sim_rows[local_idx]
            rel_row = rel_rows[local_idx]
            sim_row[global_idx] = -float("inf")
            rel_row[global_idx] = False
            n_rel = int(rel_row.sum().item())
            if n_rel <= 0:
                continue
            ranked_idx = torch.argsort(sim_row, descending=True)
            ranked_rel = rel_row[ranked_idx][:n_rel].to(torch.float32)
            prefix_all_relevant = torch.cumprod(ranked_rel, dim=0).sum().item()
            scores.append(prefix_all_relevant / float(n_rel))
        progress_log(
            output_dir,
            "query_chunk_done",
            q_start=q_start,
            q_end=q_end,
            accumulated_queries=len(scores),
        )

    if not scores:
        return 0.0, 0
    return float(sum(scores) / len(scores)), len(scores)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VPRTempo features under VIPeR-style TartanAir protocol.")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dims", default="56,56")
    parser.add_argument("--patches", type=int, default=7)
    parser.add_argument("--relevance-threshold", type=float, default=0.5)
    parser.add_argument("--sim-chunk-size", type=int, default=256)
    parser.add_argument("--gt-chunk-size", type=int, default=32)
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--log-every-samples", type=int, default=500)
    parser.add_argument("--log-every-gallery-chunks", type=int, default=8)
    parser.add_argument("--save-matrices", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.dims = [int(x) for x in args.dims.split(",")]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_eval_device(args.eval_device)
    progress_log(output_dir, "run_start", eval_device=str(device), scene_root=args.scene_root)

    embeddings, probe_points, depths_all, poses_all, intrinsics_all, seq_names = build_embeddings_and_metadata(args)
    r100p, valid_queries = evaluate_streaming_r100p(
        output_dir=output_dir,
        embeddings=embeddings,
        probe_points=probe_points,
        depths_all=depths_all,
        poses_all=poses_all,
        intrinsics_all=intrinsics_all,
        threshold=args.relevance_threshold,
        sim_chunk_size=args.sim_chunk_size,
        gt_chunk_size=args.gt_chunk_size,
        device=device,
        log_every_gallery_chunks=args.log_every_gallery_chunks,
    )

    if args.save_matrices:
        similarity = pairwise_cosine_chunked(embeddings, chunk_size=args.sim_chunk_size).numpy().astype(np.float32)
        np.save(output_dir / "similarity.npy", similarity)

    result = {
        "scene_root": str(Path(args.scene_root).resolve()).replace("\\", "/"),
        "model_path": str(Path(args.model_path).resolve()).replace("\\", "/"),
        "frame_count": int(embeddings.shape[0]),
        "valid_queries": int(valid_queries),
        "relevance_threshold": float(args.relevance_threshold),
        "R@100P": float(r100p),
        "sequence_count": int(len(sorted(set(seq_names)))),
        "sequences": sorted(set(seq_names)),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress_log(output_dir, "run_done", r100p=float(r100p), valid_queries=int(valid_queries))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


