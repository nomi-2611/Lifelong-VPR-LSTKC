from __future__ import absolute_import, print_function

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.datasets.lreid_dataset.datasets as datasets
from src.datasets.lreid_dataset.datasets.get_data_loaders import get_test_loader, get_data
from src.models.feature_extraction import extract_cnn_feature
from src.models.reid_models.layers import DataParallel
from src.models.reid_models.resnet import make_model
from src.utils.reid_utils.serialization import load_checkpoint, copy_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Export LSTKC place query/gallery embeddings.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dataset-names', nargs='+', default=['nordland_place', 'robotcar_place', 'tart_place'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--eval-query-limit', type=int, default=None)
    parser.add_argument('--eval-gallery-limit', type=int, default=None)
    parser.add_argument('--feature-kind', type=str, default='projected',
                        choices=['projected', 'raw', 'bn'],
                        help='Which feature to export: projected global feature, raw VPRTempo response, or BN feature.')

    parser.add_argument('--MODEL', type=str, default='vprtempo_snn', choices=['50x', 'snn_tiny', 'vprtempo_snn'])
    parser.add_argument('--num-instances', type=int, default=4)
    parser.add_argument('--place-preprocess-mode', type=str, default='auto', choices=['auto', 'reid', 'vprtempo'])
    parser.add_argument('--place-num-instances', type=int, default=2)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--vprtempo-preprocess-cache', action='store_true')
    parser.add_argument('--vprtempo-preprocess-cache-dir', type=str, default=None)

    parser.add_argument('--snn-input-h', type=int, default=56)
    parser.add_argument('--snn-input-w', type=int, default=56)
    parser.add_argument('--snn-hidden-dim', type=int, default=1024)
    parser.add_argument('--snn-embed-dim', type=int, default=512)
    parser.add_argument('--snn-timesteps', type=int, default=4)
    parser.add_argument('--snn-dropout', type=float, default=0.1)

    parser.add_argument('--vprtempo-model-path', type=str, default=None)
    parser.add_argument('--vprtempo-input-h', type=int, default=56)
    parser.add_argument('--vprtempo-input-w', type=int, default=56)
    parser.add_argument('--vprtempo-patches', type=int, default=7)
    parser.add_argument('--vprtempo-embed-dim', type=int, default=512)
    parser.add_argument('--vprtempo-dropout', type=float, default=0.1)
    parser.add_argument('--vprtempo-freeze-mode', type=str, default='frozen', choices=['frozen', 'trainable'])
    return parser.parse_args()


def _configure_loader_runtime(args):
    get_data._place_preprocess_mode = args.place_preprocess_mode
    get_data._model_name = args.MODEL
    get_data._vprtempo_input_h = args.vprtempo_input_h
    get_data._vprtempo_input_w = args.vprtempo_input_w
    get_data._vprtempo_patches = args.vprtempo_patches
    get_data._place_num_instances = args.place_num_instances
    get_data._persistent_workers = args.persistent_workers
    get_data._prefetch_factor = args.prefetch_factor
    get_data._vprtempo_preprocess_cache = args.vprtempo_preprocess_cache
    get_data._vprtempo_cache_dir = args.vprtempo_preprocess_cache_dir


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _extract_model_feature(model, imgs, feature_kind):
    imgs = imgs.cuda(non_blocking=True)
    if feature_kind == 'projected':
        return extract_cnn_feature(model, imgs).float()

    base_model = _unwrap_model(model)
    if feature_kind == 'raw':
        if not hasattr(base_model, 'extract_raw_feature'):
            raise AttributeError('feature-kind=raw requires a model with extract_raw_feature()')
        outputs = base_model.extract_raw_feature(imgs)
        return outputs.float().detach().cpu()

    outputs = base_model(imgs, get_all_feat=True)
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError('feature-kind=bn requires get_all_feat=True to return BN features')
    return outputs[1].float().detach().cpu()


def _extract_feature_map(model, data_loader, feature_kind):
    model.eval()
    feature_map = OrderedDict()
    start = time.time()
    with torch.no_grad():
        for imgs, fnames, pids, cids, domains in data_loader:
            outputs = _extract_model_feature(model, imgs, feature_kind)
            for fname, output in zip(fnames, outputs):
                feature_map[fname] = output.float().view(-1).cpu().numpy().astype(np.float32)
    print('Export phase [feature_extraction:{}]: {:.2f}s, features={}'.format(
        feature_kind, time.time() - start, int(len(feature_map))))
    return feature_map


def _stack_embeddings(feature_map, entries):
    payload = []
    for fname, _, _, _ in entries:
        if fname not in feature_map:
            raise KeyError('Missing extracted feature for {}'.format(fname))
        payload.append(feature_map[fname])
    return np.stack(payload, axis=0).astype(np.float32)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.vprtempo_preprocess_cache and not args.vprtempo_preprocess_cache_dir:
        args.vprtempo_preprocess_cache_dir = osp.join(args.output_dir, 'vprtempo_preprocess_cache')

    _configure_loader_runtime(args)

    first_dataset = datasets.create(args.dataset_names[0], args.data_dir)
    model = make_model(args, num_class=first_dataset.num_train_pids, camera_num=0, view_num=0)
    model.cuda()
    model = DataParallel(model)

    checkpoint = load_checkpoint(args.checkpoint)
    copy_state_dict(checkpoint['state_dict'], model)

    manifest = OrderedDict()
    summary = OrderedDict()

    for dataset_name in args.dataset_names:
        dataset = datasets.create(dataset_name, args.data_dir)
        query = dataset.query[:args.eval_query_limit] if args.eval_query_limit is not None else dataset.query
        gallery = dataset.gallery[:args.eval_gallery_limit] if args.eval_gallery_limit is not None else dataset.gallery
        testset = list(set(query) | set(gallery))
        loader = get_test_loader(dataset, 256, 128, args.batch_size, args.workers, testset=testset)

        print('Exporting embeddings for {} ({})'.format(dataset_name, args.feature_kind))
        feature_map = _extract_feature_map(model, loader, args.feature_kind)
        query_embeddings = _stack_embeddings(feature_map, query)
        gallery_embeddings = _stack_embeddings(feature_map, gallery)

        query_path = osp.abspath(osp.join(args.output_dir, '{}_query_embeddings.npy'.format(dataset_name))).replace('\\', '/')
        gallery_path = osp.abspath(osp.join(args.output_dir, '{}_gallery_embeddings.npy'.format(dataset_name))).replace('\\', '/')
        np.save(query_path, query_embeddings)
        np.save(gallery_path, gallery_embeddings)

        manifest[dataset_name] = {
            'query_embeddings': query_path,
            'gallery_embeddings': gallery_path,
            'metric': 'cosine',
            'normalized': True,
            'feature_kind': args.feature_kind,
        }
        summary[dataset_name] = {
            'query_count': int(query_embeddings.shape[0]),
            'gallery_count': int(gallery_embeddings.shape[0]),
            'dim': int(query_embeddings.shape[1]),
        }

    manifest_path = osp.join(args.output_dir, 'embedding_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    summary_path = osp.join(args.output_dir, 'embedding_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({
        'manifest_path': osp.abspath(manifest_path).replace('\\', '/'),
        'summary_path': osp.abspath(summary_path).replace('\\', '/'),
        'datasets': args.dataset_names,
    }, indent=2))


if __name__ == '__main__':
    main()
