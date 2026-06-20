from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.datasets.lreid_dataset.datasets as lstkc_datasets


class ResidualLinearAdapter(nn.Module):
    def __init__(self, dim: int, beta: float):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.beta * self.proj(x), dim=1)


class DatasetAwareResidualAdapter(nn.Module):
    def __init__(self, dim: int, beta: float, dataset_aware_tart: bool = False):
        super().__init__()
        self.shared = ResidualLinearAdapter(dim, beta)
        self.dataset_aware_tart = dataset_aware_tart
        self.tart = ResidualLinearAdapter(dim, beta) if dataset_aware_tart else None

    def forward_dataset(self, dataset_name: str, x: torch.Tensor) -> torch.Tensor:
        if self.dataset_aware_tart and dataset_name == "tart_place":
            return self.tart(x)
        return self.shared(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a residual linear adapter on frozen VPRTempo embeddings.")
    parser.add_argument("--manifest", default=None, help="Legacy single embedding manifest used for both training and evaluation")
    parser.add_argument("--train-manifest", default=None, help="Embedding manifest used for adapter training")
    parser.add_argument("--eval-manifest", default=None, help="Embedding manifest used for adapter evaluation/export")
    parser.add_argument("--data-root", required=True, help="LSTKC place dataset root")
    parser.add_argument("--dataset-names", nargs="+", default=["nordland_place", "robotcar_place", "tart_place"])
    parser.add_argument("--dataset-weights", nargs="+", type=float, default=[1.0, 1.0, 2.0])
    parser.add_argument("--beta", type=float, required=True, help="Residual adapter scale")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--gallery-loss-weight", type=float, default=0.5)
    parser.add_argument("--objective", choices=["prototype", "contrastive"], default="contrastive")
    parser.add_argument("--positive-radius", type=int, default=2, help="Neighbor radius within a scene/camera treated as positive")
    parser.add_argument("--preserve-weight", type=float, default=0.1, help="Keep adapted embeddings close to the frozen source embeddings")
    parser.add_argument("--hard-negative-weight", type=float, default=0.5,
                        help="additional weight for hardest-negative ranking loss on top of contrastive training")
    parser.add_argument("--hard-negative-margin", type=float, default=0.05,
                        help="margin enforcing positives to outrank hardest negatives")
    parser.add_argument("--tart-hard-negative-topk", type=int, default=5,
                        help="for Tart, also include the top-k most similar incorrect locations as hard negatives")
    parser.add_argument("--tart-hard-negative-radius", type=int, default=20,
                        help="for Tart, include nearby-but-outside-positive-radius samples within this radius as hard negatives")
    parser.add_argument("--dataset-aware-tart", action="store_true",
                        help="use a separate residual adapter branch for Tart while keeping a shared branch for other datasets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_manifest(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_dataset_split(data_root: str, dataset_name: str, split_name: str, expected_len: int):
    dataset = lstkc_datasets.create(dataset_name, data_root)
    split = list(getattr(dataset, split_name))
    if len(split) < expected_len:
        raise ValueError(
            "{} {} split shorter than embeddings: {} < {}".format(dataset_name, split_name, len(split), expected_len)
        )
    return split[:expected_len]


def _load_split_embeddings_from_item(item: dict, split_name: str):
    key = f"{split_name}_embeddings"
    if item.get(key) is None:
        raise KeyError(f"Manifest item does not contain {key}")
    return np.asarray(np.load(item[key]), dtype=np.float32)


def build_train_bundle(data_root: str, dataset_name: str, item: dict):
    train = _load_split_embeddings_from_item(item, "train")
    train_split = load_dataset_split(data_root, dataset_name, "train", train.shape[0])
    train_labels = np.asarray([int(sample[1]) for sample in train_split], dtype=np.int64)
    train_camids = np.asarray([int(sample[2]) for sample in train_split], dtype=np.int64)
    local_indices = np.zeros(train.shape[0], dtype=np.int64)
    per_cam_count = {}
    for idx, (_, _, camid, _) in enumerate(train_split):
        per_cam_count.setdefault(camid, 0)
        local_indices[idx] = per_cam_count[camid]
        per_cam_count[camid] += 1
    return {
        "train_embeddings": torch.from_numpy(train),
        "train_labels": torch.from_numpy(train_labels),
        "train_camids": torch.from_numpy(train_camids),
        "train_local_indices": torch.from_numpy(local_indices),
        "train_path": item["train_embeddings"],
    }


def build_eval_bundle(data_root: str, dataset_name: str, item: dict):
    query = _load_split_embeddings_from_item(item, "query")
    gallery = _load_split_embeddings_from_item(item, "gallery")
    query_split = load_dataset_split(data_root, dataset_name, "query", query.shape[0])
    gallery_split = load_dataset_split(data_root, dataset_name, "gallery", gallery.shape[0])
    query_labels = np.asarray([int(sample[1]) for sample in query_split], dtype=np.int64)
    gallery_labels = np.asarray([int(sample[1]) for sample in gallery_split], dtype=np.int64)
    return {
        "query_embeddings": torch.from_numpy(query),
        "gallery_embeddings": torch.from_numpy(gallery),
        "query_labels": torch.from_numpy(query_labels),
        "gallery_labels": torch.from_numpy(gallery_labels),
        "query_path": item["query_embeddings"],
        "gallery_path": item["gallery_embeddings"],
    }


def compute_label_prototypes(z_embeddings: torch.Tensor, labels: torch.Tensor):
    unique_labels = torch.unique(labels, sorted=True)
    label_to_index = {int(label.item()): idx for idx, label in enumerate(unique_labels)}
    proto = []
    for label in unique_labels:
        mask = labels == label
        proto.append(z_embeddings[mask].mean(dim=0))
    proto = torch.stack(proto, dim=0)
    proto = F.normalize(proto, dim=1)
    target_index = torch.tensor([label_to_index[int(x.item())] for x in labels], device=z_embeddings.device)
    return proto, label_to_index, target_index


def map_query_targets(query_labels: torch.Tensor, label_to_index: dict[int, int], device: torch.device):
    return torch.tensor([label_to_index[int(x.item())] for x in query_labels], dtype=torch.long, device=device)


def build_positive_mask(labels: torch.Tensor, camids: torch.Tensor, local_indices: torch.Tensor, positive_radius: int):
    same_label = labels[:, None] == labels[None, :]
    same_cam = camids[:, None] == camids[None, :]
    local_gap = (local_indices[:, None] - local_indices[None, :]).abs()
    nearby = same_cam & (local_gap <= positive_radius)
    positive_mask = same_label | nearby
    positive_mask.fill_diagonal_(False)
    return positive_mask


def supervised_contrastive_loss(z_embeddings: torch.Tensor, positive_mask: torch.Tensor, temperature: float):
    logits = torch.mm(z_embeddings, z_embeddings.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    valid_mask = ~self_mask
    exp_logits = torch.exp(logits) * valid_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_mask = positive_mask & valid_mask
    positive_count = positive_mask.sum(dim=1)
    valid_rows = positive_count > 0
    if not torch.any(valid_rows):
        return logits.new_tensor(0.0), 0.0

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    loss = -mean_log_prob_pos[valid_rows].mean()

    with torch.no_grad():
        positive_sim = torch.mm(z_embeddings, z_embeddings.t())[positive_mask]
        positive_sim_mean = float(positive_sim.mean().item()) if positive_sim.numel() > 0 else 0.0
    return loss, positive_sim_mean


def _build_tart_specific_negative_mask(similarity: torch.Tensor, positive_mask: torch.Tensor,
                                       camids: torch.Tensor, local_indices: torch.Tensor,
                                       positive_radius: int, topk: int, hard_radius: int):
    device = similarity.device
    n = similarity.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    negative_mask = (~positive_mask) & (~self_mask)
    same_cam = camids[:, None] == camids[None, :]
    local_gap = (local_indices[:, None] - local_indices[None, :]).abs()
    nearby_wrong = same_cam & (local_gap > positive_radius) & (local_gap <= hard_radius) & negative_mask

    neg_inf = torch.full_like(similarity, float("-inf"))
    sim_for_topk = torch.where(negative_mask, similarity, neg_inf)
    k = min(max(int(topk), 1), max(n - 1, 1))
    topk_idx = torch.topk(sim_for_topk, k=k, dim=1).indices
    topk_mask = torch.zeros_like(negative_mask)
    row_ids = torch.arange(n, device=device).unsqueeze(1).expand_as(topk_idx)
    topk_mask[row_ids, topk_idx] = True
    topk_mask = topk_mask & negative_mask
    return nearby_wrong | topk_mask


def hard_negative_ranking_loss(z_embeddings: torch.Tensor, positive_mask: torch.Tensor, margin: float,
                               negative_mask_override: torch.Tensor = None):
    similarity = torch.mm(z_embeddings, z_embeddings.t())
    self_mask = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    positive_mask = positive_mask & (~self_mask)
    if negative_mask_override is None:
        negative_mask = (~positive_mask) & (~self_mask)
    else:
        negative_mask = negative_mask_override & (~positive_mask) & (~self_mask)

    positive_count = positive_mask.sum(dim=1)
    valid_rows = positive_count > 0
    if not torch.any(valid_rows):
        return similarity.new_tensor(0.0), 0.0, 0.0

    neg_inf = torch.full_like(similarity, float("-inf"))
    pos_masked = torch.where(positive_mask, similarity, neg_inf)
    neg_masked = torch.where(negative_mask, similarity, neg_inf)

    hardest_positive = pos_masked.max(dim=1)[0]
    hardest_negative = neg_masked.max(dim=1)[0]
    ranking = F.relu(margin + hardest_negative - hardest_positive)
    ranking = ranking[valid_rows]
    if ranking.numel() == 0:
        return similarity.new_tensor(0.0), 0.0, 0.0

    with torch.no_grad():
        hard_pos_mean = float(hardest_positive[valid_rows].mean().item())
        hard_neg_mean = float(hardest_negative[valid_rows].mean().item())
    return ranking.mean(), hard_pos_mean, hard_neg_mean


def train_adapter(train_bundles: OrderedDict[str, dict], dataset_weights: list[float], beta: float, epochs: int,
                  lr: float, weight_decay: float, temperature: float, gallery_loss_weight: float,
                  objective: str, positive_radius: int, preserve_weight: float,
                  hard_negative_weight: float, hard_negative_margin: float,
                  tart_hard_negative_topk: int, tart_hard_negative_radius: int,
                  dataset_aware_tart: bool):
    sample_tensor = next(iter(train_bundles.values()))["train_embeddings"]
    dim = int(sample_tensor.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = DatasetAwareResidualAdapter(dim, beta, dataset_aware_tart=dataset_aware_tart).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    weight_map = {
        dataset_name: float(dataset_weights[idx])
        for idx, dataset_name in enumerate(train_bundles.keys())
    }

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        epoch_stats = {}
        for dataset_name, bundle in train_bundles.items():
            base_train = bundle["train_embeddings"].to(device)
            z_train = adapter.forward_dataset(dataset_name, base_train)
            if objective == "prototype":
                proto, _, train_targets = compute_label_prototypes(
                    z_train, bundle["train_labels"].to(device)
                )
                train_logits = torch.mm(z_train, proto.t()) / temperature
                main_loss = F.cross_entropy(train_logits, train_targets)
                with torch.no_grad():
                    score = (train_logits.argmax(dim=1) == train_targets).float().mean().item()
                epoch_stats[dataset_name] = {
                    "top1": float(score),
                }
            else:
                positive_mask = build_positive_mask(
                    bundle["train_labels"].to(device),
                    bundle["train_camids"].to(device),
                    bundle["train_local_indices"].to(device),
                    positive_radius=positive_radius,
                )
                main_loss, positive_sim_mean = supervised_contrastive_loss(
                    z_train, positive_mask, temperature
                )
                negative_override = None
                if dataset_name == "tart_place":
                    negative_override = _build_tart_specific_negative_mask(
                        torch.mm(z_train, z_train.t()),
                        positive_mask,
                        bundle["train_camids"].to(device),
                        bundle["train_local_indices"].to(device),
                        positive_radius=positive_radius,
                        topk=tart_hard_negative_topk,
                        hard_radius=tart_hard_negative_radius,
                    )
                hard_neg_loss, hard_pos_mean, hard_neg_mean = hard_negative_ranking_loss(
                    z_train, positive_mask, hard_negative_margin,
                    negative_mask_override=negative_override,
                )
                main_loss = main_loss + hard_negative_weight * hard_neg_loss
                epoch_stats[dataset_name] = {
                    "positive_sim_mean": float(positive_sim_mean),
                    "hard_positive_mean": float(hard_pos_mean),
                    "hard_negative_mean": float(hard_neg_mean),
                    "hard_negative_loss": float(hard_neg_loss.item()),
                    "hard_negative_mode": "tart_specific" if dataset_name == "tart_place" else "default",
                }

            preserve_loss = 1.0 - (z_train * F.normalize(base_train, dim=1)).sum(dim=1).mean()
            dataset_loss = main_loss + preserve_weight * preserve_loss
            weighted_loss = weight_map[dataset_name] * dataset_loss
            total_loss = total_loss + weighted_loss

            with torch.no_grad():
                epoch_stats[dataset_name]["loss"] = float(dataset_loss.item())
                epoch_stats[dataset_name]["preserve_loss"] = float(preserve_loss.item())

        total_loss.backward()
        optimizer.step()

        history.append({
            "epoch": epoch + 1,
            "loss": float(total_loss.item()),
            "per_dataset": epoch_stats,
        })

    return adapter.cpu(), history


def transform_eval_bundle(adapter: DatasetAwareResidualAdapter, dataset_name: str, bundle: dict):
    with torch.no_grad():
        q = adapter.forward_dataset(dataset_name, bundle["query_embeddings"]).cpu().numpy().astype(np.float32)
        g = adapter.forward_dataset(dataset_name, bundle["gallery_embeddings"]).cpu().numpy().astype(np.float32)
    return q, g


def tart_diag_stats(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray):
    similarity = np.matmul(query_embeddings, gallery_embeddings.T)
    n = min(similarity.shape[0], similarity.shape[1])
    diag = similarity[np.arange(n), np.arange(n)]
    top1 = similarity.argmax(axis=1)
    row_max = similarity.max(axis=1)
    row_second = np.partition(similarity, -2, axis=1)[:, -2]
    return {
        "diag_mean": float(diag.mean()),
        "row_max_mean": float(row_max.mean()),
        "gap_mean": float((row_max - row_second).mean()),
        "top1_diag_rate": float((top1[:n] == np.arange(n)).mean()),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.train_manifest is None and args.eval_manifest is None:
        if args.manifest is None:
            raise ValueError("Provide --manifest or both --train-manifest and --eval-manifest")
        args.train_manifest = args.manifest
        args.eval_manifest = args.manifest
    elif args.train_manifest is None or args.eval_manifest is None:
        raise ValueError("--train-manifest and --eval-manifest must be provided together")

    if len(args.dataset_weights) != len(args.dataset_names):
        raise ValueError("--dataset-weights must match --dataset-names length")

    train_manifest = load_manifest(args.train_manifest)
    eval_manifest = load_manifest(args.eval_manifest)
    train_bundles = OrderedDict()
    eval_bundles = OrderedDict()
    for dataset_name in args.dataset_names:
        if dataset_name not in train_manifest:
            raise KeyError("Dataset {} missing from train manifest {}".format(dataset_name, args.train_manifest))
        if dataset_name not in eval_manifest:
            raise KeyError("Dataset {} missing from eval manifest {}".format(dataset_name, args.eval_manifest))
        train_bundles[dataset_name] = build_train_bundle(args.data_root, dataset_name, train_manifest[dataset_name])
        eval_bundles[dataset_name] = build_eval_bundle(args.data_root, dataset_name, eval_manifest[dataset_name])

    adapter, history = train_adapter(
        train_bundles=train_bundles,
        dataset_weights=args.dataset_weights,
        beta=args.beta,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        gallery_loss_weight=args.gallery_loss_weight,
        objective=args.objective,
        positive_radius=args.positive_radius,
        preserve_weight=args.preserve_weight,
        hard_negative_weight=args.hard_negative_weight,
        hard_negative_margin=args.hard_negative_margin,
        tart_hard_negative_topk=args.tart_hard_negative_topk,
        tart_hard_negative_radius=args.tart_hard_negative_radius,
        dataset_aware_tart=args.dataset_aware_tart,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_out = {}
    tart_stats = None
    for dataset_name, bundle in eval_bundles.items():
        q, g = transform_eval_bundle(adapter, dataset_name, bundle)
        q_path = osp.join(args.output_dir, "{}_query_embeddings.npy".format(dataset_name))
        g_path = osp.join(args.output_dir, "{}_gallery_embeddings.npy".format(dataset_name))
        np.save(q_path, q)
        np.save(g_path, g)
        manifest_out[dataset_name] = {
            "query_embeddings": osp.abspath(q_path).replace("\\", "/"),
            "gallery_embeddings": osp.abspath(g_path).replace("\\", "/"),
            "metric": "cosine",
            "normalized": True,
        }
        if dataset_name == "tart_place":
            tart_stats = tart_diag_stats(q, g)

    manifest_path = osp.join(args.output_dir, "adapter_embedding_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2)

    state_path = osp.join(args.output_dir, "adapter_state.pth")
    torch.save({
        "beta": args.beta,
        "state_dict": adapter.state_dict(),
        "dataset_names": args.dataset_names,
        "dataset_weights": args.dataset_weights,
    }, state_path)

    summary = {
        "train_manifest": osp.abspath(args.train_manifest).replace("\\", "/"),
        "eval_manifest": osp.abspath(args.eval_manifest).replace("\\", "/"),
        "output_manifest": osp.abspath(manifest_path).replace("\\", "/"),
        "adapter_state": osp.abspath(state_path).replace("\\", "/"),
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "gallery_loss_weight": args.gallery_loss_weight,
        "objective": args.objective,
        "positive_radius": args.positive_radius,
        "preserve_weight": args.preserve_weight,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_negative_margin": args.hard_negative_margin,
        "tart_hard_negative_topk": args.tart_hard_negative_topk,
        "tart_hard_negative_radius": args.tart_hard_negative_radius,
        "dataset_aware_tart": bool(args.dataset_aware_tart),
        "dataset_weights": {
            dataset_name: args.dataset_weights[idx]
            for idx, dataset_name in enumerate(args.dataset_names)
        },
        "final_history": history[-1] if history else {},
        "tart_diag_stats": tart_stats,
    }
    summary_path = osp.join(args.output_dir, "adapter_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
