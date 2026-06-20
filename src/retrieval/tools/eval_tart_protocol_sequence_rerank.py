from __future__ import absolute_import, print_function

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eval_tart_protocol_with_vprtempo import compute_metrics


def _load_paths(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _load_protocol(protocol_dir, scene):
    protocol_dir = Path(protocol_dir)
    query_paths = _load_paths(protocol_dir / '{}_query_paths.json'.format(scene))
    gallery_paths = _load_paths(protocol_dir / '{}_gallery_paths.json'.format(scene))
    gt = np.asarray(np.load(protocol_dir / '{}_gt.npy'.format(scene)), dtype=np.uint8)
    return query_paths, gallery_paths, gt


def _sequence_groups(paths):
    groups = []
    start = 0
    def route_key(path):
        return str(Path(path).parent.parent)
    while start < len(paths):
        key = route_key(paths[start])
        end = start + 1
        while end < len(paths) and route_key(paths[end]) == key:
            end += 1
        groups.append((start, end))
        start = end
    return groups


def _group_bounds(paths):
    starts = np.zeros((len(paths),), dtype=np.int64)
    ends = np.zeros((len(paths),), dtype=np.int64)
    for start, end in _sequence_groups(paths):
        starts[start:end] = start
        ends[start:end] = end
    return starts, ends


def _normalize(array):
    tensor = torch.as_tensor(np.asarray(array, dtype=np.float32)).contiguous()
    return F.normalize(tensor, p=2, dim=1)


def _rerank_similarity(query_embeddings, gallery_embeddings, query_paths, gallery_paths,
                       radius=30, candidate_topk=1000, query_block_size=32, candidate_chunk_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q = _normalize(query_embeddings).to(device)
    g = _normalize(gallery_embeddings).to(device)
    num_queries = q.size(0)
    num_gallery = g.size(0)
    candidate_topk = min(max(10, int(candidate_topk)), int(num_gallery))
    q_starts, q_ends = _group_bounds(query_paths)
    g_starts, g_ends = _group_bounds(gallery_paths)
    output = np.full((int(num_queries), int(num_gallery)), -np.inf, dtype=np.float32)

    for q_start in range(0, int(num_queries), int(query_block_size)):
        q_end = min(q_start + int(query_block_size), int(num_queries))
        q_block = q[q_start:q_end]
        base_sim, candidates = torch.topk(torch.mm(q_block, g.t()), k=candidate_topk, dim=1, largest=True, sorted=True)
        block_size = q_end - q_start
        candidates_cpu = candidates.detach().cpu().numpy()

        score_sum = torch.zeros_like(base_sim)
        score_count = torch.zeros_like(base_sim)
        q_indices = torch.arange(q_start, q_end, device=device, dtype=torch.long)
        q_group_start = torch.as_tensor(q_starts[q_start:q_end], device=device, dtype=torch.long)
        q_group_end = torch.as_tensor(q_ends[q_start:q_end], device=device, dtype=torch.long)
        g_group_start = torch.as_tensor(
            g_starts[candidates_cpu.reshape(-1)].reshape(block_size, candidate_topk),
            device=device,
            dtype=torch.long,
        )
        g_group_end = torch.as_tensor(
            g_ends[candidates_cpu.reshape(-1)].reshape(block_size, candidate_topk),
            device=device,
            dtype=torch.long,
        )
        for offset in range(-int(radius), int(radius) + 1):
            shifted_q = q_indices + int(offset)
            valid_q = (shifted_q >= q_group_start) & (shifted_q < q_group_end)
            safe_q = shifted_q.clamp(0, int(num_queries) - 1)
            shifted_q_feat = q[safe_q]
            for cand_start in range(0, candidate_topk, int(candidate_chunk_size)):
                cand_end = min(cand_start + int(candidate_chunk_size), candidate_topk)
                shifted_g = candidates[:, cand_start:cand_end] + int(offset)
                valid_g = (
                    (shifted_g >= g_group_start[:, cand_start:cand_end]) &
                    (shifted_g < g_group_end[:, cand_start:cand_end])
                )
                valid = valid_q.reshape(-1, 1) & valid_g
                if not bool(valid.any().item()):
                    continue
                safe_g = shifted_g.clamp(0, int(num_gallery) - 1)
                shifted_g_feat = g[safe_g.reshape(-1)].reshape(block_size, cand_end - cand_start, -1)
                sim = (shifted_g_feat * shifted_q_feat.unsqueeze(1)).sum(dim=2)
                score_sum[:, cand_start:cand_end] += torch.where(valid, sim, torch.zeros_like(sim))
                score_count[:, cand_start:cand_end] += valid.float()
        seq_sim = torch.where(score_count > 0, score_sum / score_count.clamp_min(1.0), base_sim)
        seq_sim_cpu = seq_sim.detach().cpu().numpy().astype(np.float32)
        for row in range(block_size):
            output[q_start + row, candidates_cpu[row]] = seq_sim_cpu[row]
        if q_end == int(num_queries) or (q_end // int(query_block_size)) % 50 == 0:
            print('rerank progress {}/{}'.format(q_end, int(num_queries)))
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return output


def parse_args():
    parser = argparse.ArgumentParser(description='Sequence-rerank exported TART protocol embeddings.')
    parser.add_argument('--protocol-dir', required=True)
    parser.add_argument('--embedding-dir', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--radius', type=int, default=30)
    parser.add_argument('--candidate-topk', type=int, default=1000)
    parser.add_argument('--query-block-size', type=int, default=32)
    parser.add_argument('--candidate-chunk-size', type=int, default=64)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    query_paths, gallery_paths, gt = _load_protocol(args.protocol_dir, args.scene)
    embedding_dir = Path(args.embedding_dir)
    q = np.asarray(np.load(embedding_dir / '{}_query_embeddings.npy'.format(args.scene)), dtype=np.float32)
    g = np.asarray(np.load(embedding_dir / '{}_gallery_embeddings.npy'.format(args.scene)), dtype=np.float32)
    sim = _rerank_similarity(
        q,
        g,
        query_paths,
        gallery_paths,
        radius=args.radius,
        candidate_topk=args.candidate_topk,
        query_block_size=args.query_block_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    metrics = compute_metrics(sim, gt, topk=(1, 5, 10))
    result = {
        'scene': args.scene,
        'radius': int(args.radius),
        'candidate_topk': int(args.candidate_topk),
        **metrics,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
