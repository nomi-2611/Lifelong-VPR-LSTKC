from __future__ import print_function, absolute_import
import argparse
import json
import os
import os.path as osp
import sys
from collections import OrderedDict

from torch.backends import cudnn
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader
from configs.defaults import _C as cfg
from src.evaluation.evaluators import Evaluator
from src.evaluation.vpr_evaluators import evaluate_vpr_dataset, evaluate_vpr_embeddings, _build_positive_map
from src.retrieval.tools.eval_place_sequence_rerank import _evaluate_stream_candidate_sequence
from src.models.feature_extraction import extract_cnn_feature
from src.utils.reid_utils.logging import Logger
from src.utils.reid_utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from src.utils.reid_utils.lr_scheduler import WarmupMultiStepLR
from src.utils.reid_utils.feature_tools import *
from src.utils.reid_utils.vprtempo_stage_adapter import adapt_vprtempo_for_stage
from src.models.reid_models.layers import DataParallel
from src.models.reid_models.resnet import make_model
from src.trainers.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from src.datasets.lreid_dataset.datasets.get_data_loaders import build_data_loaders, get_test_loader
from src.datasets.lreid_dataset.datasets.get_data_loaders import _build_preprocessor, _build_transforms, _current_cfg_proxy
from src.datasets.lreid_dataset.datasets.get_data_loaders import _dataloader_kwargs
from src.utils.reid_utils.data import IterLoader
from src.retrieval.tools.Logger_results import Logger_res
def main():
    args = parser.parse_args()
    if args.vprtempo_preprocess_cache and not args.vprtempo_preprocess_cache_dir:
        args.vprtempo_preprocess_cache_dir = osp.join(args.logs_dir, 'vprtempo_preprocess_cache')

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("cudnn benchmark enabled; deterministic reproducibility is disabled")
    if args.tf32:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
            except AttributeError:
                pass
            print("tf32 enabled for CUDA matmul/cudnn")
        else:
            print("tf32 requested but CUDA is unavailable; skipping")
    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    os.makedirs(args.logs_dir, exist_ok=True)
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder) if args.test_folder else args.logs_dir
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results

    if args.task_type == 'place':
        if args.place_train_seq:
            training_set = [item.strip() for item in args.place_train_seq.split(',') if item.strip()]
        else:
            training_set = ['nordland_place', 'robotcar_place']
    elif 1 == args.setting:
        # training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']
        # training_set = ['market1501']
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc']
    else:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    if args.max_stages is not None:
        training_set = training_set[:args.max_stages]
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid'
    # testing_only_set = [x for x in all_set if x not in training_set]
    testing_only_set = [] if args.task_type == 'place' else ['market1501']
    # print("DEBUG training_set =", training_set)
    # print("DEBUG testing_only_set =", testing_only_set)

    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)
    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0)
    model.cuda()
    model = DataParallel(model)
    pending_evaluate = args.evaluate and not args.test_folder
    if pending_evaluate:
        args.evaluate = False
    writer = SummaryWriter(log_dir=args.logs_dir)
    if args.evaluate and not args.test_folder:
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets) - 1, logger_res=logger_res, args=args)
        return
    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model
        copy_state_dict(checkpoint['state_dict'], model)
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)

                         
            best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, step + 1)
            model = linear_combination(args, model, model_old, best_alpha)
            save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1, logger_res=logger_res, args=args)

        exit(0)


    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    if pending_evaluate:
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets) - 1, logger_res=logger_res, args=args)
        return
   
    if args.MODEL in ['50x']:
        out_channel = 2048
    elif args.MODEL == 'snn_tiny':
        out_channel = args.snn_embed_dim
    elif args.MODEL == 'vprtempo_snn':
        out_channel = args.vprtempo_embed_dim
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")


    for set_index in range(0, len(training_set)):       
        model_old = copy.deepcopy(model)
        if args.MODEL == 'vprtempo_snn' and args.vprtempo_stage_adapt:
            dataset_stage = all_train_sets[set_index][0]
            stage_name = all_train_sets[set_index][-1]
            adapted_path = adapt_vprtempo_for_stage(
                args=args,
                dataset=dataset_stage,
                stage_name=stage_name,
                stage_index=set_index,
                current_model_path=getattr(model.module, 'model_path', args.vprtempo_model_path),
            )
            model.module.reload_base_from_model_path(adapted_path, freeze_mode=args.vprtempo_freeze_mode)
        model = train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer,logger_res=logger_res, stage_reference_model=model_old)
        if set_index>0:
            best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, set_index)
            if not bool(getattr(args, 'disable_stage_model_fusion', False)):
                model = linear_combination(args, model, model_old, best_alpha)
            if set_index == len(training_set) - 1:
                test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, args=args)
        if len(training_set) == 1:
            test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, args=args)
    print('finished')
def get_normal_affinity(x,Norm=100):

    from src.knowledge.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix


def _subsample_feature_list(features, limit):
    if limit is None or limit <= 0 or len(features) <= limit:
        return features
    indices = np.linspace(0, len(features) - 1, num=limit, dtype=int)
    return [features[idx] for idx in indices]


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _extract_raw_tensor(model, inputs):
    base_model = _unwrap_model(model)
    if not hasattr(base_model, 'extract_raw_feature'):
        raise AttributeError('raw memory distill requires extract_raw_feature()')
    return base_model.extract_raw_feature(inputs)


class MultiRootMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        from PIL import Image
        (fpath, pid, camid, domain), root, _ = self.entries[index]
        image_path = fpath if osp.isabs(fpath) else osp.join(root, fpath)
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fpath, pid, camid, domain


def _build_raw_memory_loader(dataset, name, args):
    memory_size = int(getattr(args, 'place_raw_memory_size', 0) or 0)
    if memory_size <= 0:
        return None
    train_entries = list(sorted(dataset.train))
    if not train_entries:
        return None
    if len(train_entries) > memory_size:
        indices = np.linspace(0, len(train_entries) - 1, num=memory_size, dtype=int)
        train_entries = [train_entries[int(index)] for index in indices]
    memory_batch_size = min(
        max(1, int(getattr(args, 'place_raw_memory_batch_size', 128) or 128)),
        len(train_entries),
    )
    _, test_transformer = _build_transforms(
        cfg=_current_cfg_proxy(),
        dataset_name=name,
        height=args.height,
        width=args.width,
    )
    memory_preprocessor = _build_preprocessor(
        train_entries,
        dataset.images_dir,
        test_transformer,
        name,
        'raw_memory',
    )
    memory_workers = int(getattr(args, 'place_raw_memory_workers', -1))
    if memory_workers < 0:
        memory_workers = args.eval_workers if getattr(args, 'eval_workers', None) is not None else args.workers
    loader = DataLoader(
        memory_preprocessor,
        batch_size=memory_batch_size,
        num_workers=memory_workers,
        shuffle=True,
        drop_last=True,
        **_dataloader_kwargs(memory_workers),
    )
    length = max(1, len(loader))
    print('Raw memory loader [{}]: samples={}, batch_size={}, iters={}'.format(
        name, len(train_entries), memory_batch_size, length))
    return IterLoader(loader, length=length)


def _build_multi_raw_memory_loader(dataset_items, args):
    memory_size = int(getattr(args, 'place_raw_memory_size', 0) or 0)
    if memory_size <= 0 or not dataset_items:
        return None
    entries_with_roots = []
    per_dataset = max(1, memory_size // len(dataset_items))
    for dataset, name in dataset_items:
        train_entries = list(sorted(dataset.train))
        if not train_entries:
            continue
        take = min(per_dataset, len(train_entries))
        indices = np.linspace(0, len(train_entries) - 1, num=take, dtype=int)
        for index in indices:
            entries_with_roots.append((train_entries[int(index)], dataset.images_dir, name))
    if not entries_with_roots:
        return None

    _, test_transformer = _build_transforms(
        cfg=_current_cfg_proxy(),
        dataset_name=dataset_items[-1][1],
        height=args.height,
        width=args.width,
    )

    memory_batch_size = min(
        max(1, int(getattr(args, 'place_raw_memory_batch_size', 128) or 128)),
        len(entries_with_roots),
    )
    memory_workers = int(getattr(args, 'place_raw_memory_workers', -1))
    if memory_workers < 0:
        memory_workers = args.eval_workers if getattr(args, 'eval_workers', None) is not None else args.workers
    loader = DataLoader(
        MultiRootMemoryDataset(entries_with_roots, test_transformer),
        batch_size=memory_batch_size,
        num_workers=memory_workers,
        shuffle=True,
        drop_last=True,
        **_dataloader_kwargs(memory_workers),
    )
    length = max(1, len(loader))
    names = ','.join(name for _, name in dataset_items)
    print('Raw memory loader [multi:{}]: samples={}, batch_size={}, iters={}'.format(
        names, len(entries_with_roots), memory_batch_size, length))
    return IterLoader(loader, length=length)


def _extract_feature_map_for_offline_eval(model, data_loader, training_phase=None, feature_kind='projected'):
    model.eval()
    features = OrderedDict()
    with torch.no_grad():
        for imgs, fnames, pids, cids, domains in data_loader:
            if feature_kind == 'projected':
                outputs = extract_cnn_feature(model, imgs, training_phase=training_phase)
            else:
                imgs = imgs.cuda(non_blocking=True)
                base_model = _unwrap_model(model)
                if feature_kind == 'raw':
                    if not hasattr(base_model, 'extract_raw_feature'):
                        raise AttributeError('place offline raw eval requires extract_raw_feature()')
                    outputs = base_model.extract_raw_feature(imgs).float().detach().cpu()
                elif feature_kind == 'bn':
                    outputs_all = base_model(imgs, get_all_feat=True)
                    if not isinstance(outputs_all, (tuple, list)) or len(outputs_all) < 2:
                        raise RuntimeError('place offline BN eval requires get_all_feat=True BN output')
                    outputs = outputs_all[1].float().detach().cpu()
                else:
                    raise ValueError('Unsupported offline eval feature kind: {}'.format(feature_kind))
            for fname, output in zip(fnames, outputs):
                features[fname] = output.float().view(-1).cpu().numpy().astype(np.float32)
    return features


def _stack_feature_array(feature_map, entries):
    return np.stack([feature_map[fname] for fname, _, _, _ in entries], axis=0).astype(np.float32)


def _resolve_place_offline_eval_dir(args):
    if args.place_offline_eval_dir:
        return args.place_offline_eval_dir
    return osp.join(args.logs_dir, 'offline_eval')


def _offline_eval_single_place_dataset(model, dataset, test_loader, name, args, export_dir, feature_kind='projected'):
    eval_workers = args.eval_workers if getattr(args, 'eval_workers', None) is not None else args.workers
    query = dataset.query[:args.eval_query_limit] if args.eval_query_limit is not None else dataset.query
    gallery = dataset.gallery[:args.eval_gallery_limit] if args.eval_gallery_limit is not None else dataset.gallery
    eval_loader = get_test_loader(dataset, 256, 128, args.batch_size, eval_workers, testset=list(set(query) | set(gallery))) \
        if (args.eval_query_limit is not None or args.eval_gallery_limit is not None) else test_loader

    print('Results on {} [{}]'.format(name, feature_kind))
    print('Offline place eval export -> {}'.format(export_dir))
    feature_map = _extract_feature_map_for_offline_eval(model, eval_loader, feature_kind=feature_kind)
    query_embeddings = _stack_feature_array(feature_map, query)
    gallery_embeddings = _stack_feature_array(feature_map, gallery)

    os.makedirs(export_dir, exist_ok=True)
    suffix = '' if feature_kind == 'projected' else '_{}'.format(feature_kind)
    query_path = osp.abspath(osp.join(export_dir, '{}{}_query_embeddings.npy'.format(name, suffix))).replace('\\', '/')
    gallery_path = osp.abspath(osp.join(export_dir, '{}{}_gallery_embeddings.npy'.format(name, suffix))).replace('\\', '/')
    np.save(query_path, query_embeddings)
    np.save(gallery_path, gallery_embeddings)

    if getattr(args, 'place_sequence_rerank', False):
        positive_map, protocol_settings = _build_positive_map(
            name,
            query,
            gallery,
            protocol=args.place_vpr_protocol,
            nordland_tolerance=args.nordland_eval_tolerance,
            robotcar_tolerance=args.robotcar_eval_tolerance,
            tart_pose_threshold=args.tart_eval_pose_threshold,
        )
        rerank_results = _evaluate_stream_candidate_sequence(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            query=query,
            gallery=gallery,
            positive_map=positive_map,
            radii=[int(args.place_sequence_radius)],
            weights=[float(args.place_sequence_weight)],
            candidate_topk=int(args.place_sequence_candidate_topk),
            query_block_size=int(args.place_sequence_query_block_size),
            candidate_chunk_size=int(args.place_sequence_candidate_chunk_size),
        )
        rerank_key = 'r{}_w{:.2f}'.format(
            int(args.place_sequence_radius),
            float(args.place_sequence_weight),
        )
        rerank_item = rerank_results[rerank_key]
        R1 = float(rerank_item['R@1'])
        mAP = float(rerank_item['mAP'])
        print('VPR sequence-rerank results on {}'.format(name))
        print('Protocol: {}'.format(protocol_settings['protocol_name']))
        print('Sequence radius={}, weight={}, candidate_topk={}'.format(
            int(args.place_sequence_radius),
            float(args.place_sequence_weight),
            int(args.place_sequence_candidate_topk),
        ))
        print('VPR mAP: {:4.1%}'.format(mAP))
        print('Recall@K:')
        print('  R@{:<4}{:12.1%}'.format(1, float(rerank_item['R@1'])))
        print('  R@{:<4}{:12.1%}'.format(5, float(rerank_item['R@5'])))
        print('  R@{:<4}{:12.1%}'.format(10, float(rerank_item['R@10'])))
    else:
        R1, mAP = evaluate_vpr_embeddings(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            query=query,
            gallery=gallery,
            dataset_name=name,
            metric=True,
            nordland_tolerance=args.nordland_eval_tolerance,
            robotcar_tolerance=args.robotcar_eval_tolerance,
            tart_pose_threshold=args.tart_eval_pose_threshold,
            protocol=args.place_vpr_protocol,
        )
    manifest_item = {
        'query_embeddings': query_path,
        'gallery_embeddings': gallery_path,
        'metric': 'cosine',
        'normalized': True,
        'feature_kind': feature_kind,
    }
    return R1, mAP, manifest_item


def _resolve_place_offline_feature_kinds(args):
    feature_kind = getattr(args, 'place_offline_eval_feature', 'projected')
    if feature_kind == 'auto':
        if getattr(args, 'MODEL', None) == 'vprtempo_snn':
            return ['projected', 'raw']
        return ['projected']
    return [feature_kind]


def test_model_place_offline(model, all_train_sets, all_test_sets, set_index, logger_res=None, args=None):
    export_dir = _resolve_place_offline_eval_dir(args)
    os.makedirs(export_dir, exist_ok=True)
    feature_kinds = _resolve_place_offline_feature_kinds(args)
    manifests = OrderedDict((feature_kind, OrderedDict()) for feature_kind in feature_kinds)
    primary_feature_kind = feature_kinds[-1] if getattr(args, 'place_offline_eval_primary_feature', 'first') == 'last' else feature_kinds[0]
    primary_result = None

    def _evaluate_group(datasets_group, stop_index=None, feature_kind='projected'):
        R1_all = []
        mAP_all = []
        names = ''
        results = ''
        last_mAP = 0.0

        upper = len(datasets_group) if stop_index is None else (stop_index + 1)
        for i in range(upper):
            dataset, num_classes, train_loader, test_loader, init_loader, name = datasets_group[i]
            R1, mAP, manifest_item = _offline_eval_single_place_dataset(
                model=model,
                dataset=dataset,
                test_loader=test_loader,
                name=name,
                args=args,
                export_dir=export_dir,
                feature_kind=feature_kind,
            )
            manifests[feature_kind][name] = manifest_item
            R1_all.append(R1)
            mAP_all.append(mAP)
            names = names + name + '\t\t'
            results = results + '|{:.1f}/{:.1f}\t'.format(mAP * 100, R1 * 100)
            last_mAP = mAP

        if len(mAP_all) == 0:
            return None, None, '', '', last_mAP
        aver_mAP = torch.tensor(mAP_all).mean()
        aver_R1 = torch.tensor(R1_all).mean()
        names = names + '|Average\t|'
        results = results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100)
        return aver_R1, aver_mAP, names, results, last_mAP

    for feature_kind in feature_kinds:
        print('Offline place eval feature kind: {}'.format(feature_kind))
        aver_R1, aver_mAP, names, results, train_mAP = _evaluate_group(
            all_train_sets,
            stop_index=set_index,
            feature_kind=feature_kind,
        )
        print("Average mAP on Seen dataset [{}]: {:.1f}%".format(feature_kind, aver_mAP * 100))
        print("Average R1 on Seen dataset [{}]: {:.1f}%".format(feature_kind, aver_R1 * 100))
        print(names)
        print(results)

        aver_R1_unseen, aver_mAP_unseen, names_unseen, results_unseen, _ = _evaluate_group(
            all_test_sets,
            feature_kind=feature_kind,
        )
        if aver_mAP_unseen is not None:
            print("Average mAP on unSeen dataset [{}]: {:.1f}%".format(feature_kind, aver_mAP_unseen * 100))
            print("Average R1 on unSeen dataset [{}]: {:.1f}%".format(feature_kind, aver_R1_unseen * 100))
            print(names_unseen)
            print(results_unseen)
        else:
            print("No unseen-only datasets configured for this run.")

        manifest_name = 'embedding_manifest.json' if feature_kind == 'projected' else 'embedding_manifest_{}.json'.format(feature_kind)
        manifest_path = osp.join(export_dir, manifest_name)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifests[feature_kind], f, indent=2)
        print('Offline place eval manifest saved to {}'.format(osp.abspath(manifest_path)))

        if feature_kind == primary_feature_kind:
            primary_result = (names, results, names_unseen, results_unseen, aver_mAP_unseen, train_mAP)

    if logger_res:
        names, results, names_unseen, results_unseen, aver_mAP_unseen, train_mAP = primary_result
        logger_res.append(names)
        logger_res.append(results)
        logger_res.append(results.replace('|', '').replace('/', '\t'))
        if aver_mAP_unseen is not None:
            logger_res.append(names_unseen)
            logger_res.append(results_unseen)
            logger_res.append(results_unseen.replace('|', '').replace('/', '\t'))
    return primary_result[-1] if primary_result is not None else 0.0


def load_vprtempo_prototype_bank(manifest_path, dataset):
    if manifest_path is None:
        return None
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    item = manifest.get(dataset.dataset_name)
    if item is None:
        return None

    query_embeddings = np.asarray(np.load(item['query_embeddings']), dtype=np.float32)
    gallery_embeddings = np.asarray(np.load(item['gallery_embeddings']), dtype=np.float32)
    embeddings = []
    counts = {}

    def _accumulate(samples, feats):
        if len(samples) != feats.shape[0]:
            raise ValueError('Embedding count mismatch for {}: {} samples vs {} embeddings'.format(
                dataset.dataset_name, len(samples), feats.shape[0]))
        for (_, pid, _, _), feat in zip(samples, feats):
            pid = int(pid)
            if pid not in counts:
                counts[pid] = 0
            while len(embeddings) <= pid:
                embeddings.append(np.zeros(feats.shape[1], dtype=np.float32))
            embeddings[pid] += feat
            counts[pid] += 1

    _accumulate(dataset.query[:query_embeddings.shape[0]], query_embeddings)
    _accumulate(dataset.gallery[:gallery_embeddings.shape[0]], gallery_embeddings)

    if not embeddings:
        return None

    bank = np.stack(embeddings, axis=0)
    mask = np.zeros(bank.shape[0], dtype=np.float32)
    for pid, count in counts.items():
        if count > 0:
            bank[pid] /= float(count)
            mask[pid] = 1.0
    return {
        'bank': torch.from_numpy(bank).float(),
        'mask': torch.from_numpy(mask).float(),
        'dim': int(bank.shape[1]),
        'dataset_name': dataset.dataset_name,
    }


def _normalize_feature_path(path):
    return osp.normcase(osp.normpath(str(path)))


def load_teacher_feature_bank(manifest_path, dataset):
    if manifest_path is None:
        return None
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    item = manifest.get(dataset.dataset_name)
    if item is None:
        return None

    teacher_bank = {}

    def _accumulate(samples, feats, split_name):
        if len(samples) != feats.shape[0]:
            usable = min(len(samples), feats.shape[0])
            print('Teacher embedding count mismatch for {} {}: {} samples vs {} embeddings; using first {}'.format(
                dataset.dataset_name, split_name, len(samples), feats.shape[0], usable))
            samples = samples[:usable]
            feats = feats[:usable]
        for (path, _, _, _), feat in zip(samples, feats):
            teacher_bank[_normalize_feature_path(path)] = torch.from_numpy(feat).float()

    if 'train_embeddings' in item:
        train_embeddings = np.asarray(np.load(item['train_embeddings']), dtype=np.float32)
        _accumulate(dataset.train[:train_embeddings.shape[0]], train_embeddings, 'train')
    if 'query_embeddings' in item:
        query_embeddings = np.asarray(np.load(item['query_embeddings']), dtype=np.float32)
        _accumulate(dataset.query[:query_embeddings.shape[0]], query_embeddings, 'query')
    if 'gallery_embeddings' in item:
        gallery_embeddings = np.asarray(np.load(item['gallery_embeddings']), dtype=np.float32)
        _accumulate(dataset.gallery[:gallery_embeddings.shape[0]], gallery_embeddings, 'gallery')
    if not teacher_bank:
        return None
    return teacher_bank


def build_self_teacher_feature_bank(model, data_loader, limit=None):
    if limit is not None and int(limit) <= 0:
        limit = None
    teacher_bank = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for imgs, fnames, _, _, _ in data_loader:
            imgs = imgs.cuda(non_blocking=True)
            if hasattr(model.module, 'extract_raw_feature'):
                feats = model.module.extract_raw_feature(imgs)
            else:
                feats = model(imgs)
            feats = torch.nn.functional.normalize(feats.float(), dim=1).cpu()
            for fname, feat in zip(fnames, feats):
                teacher_bank[_normalize_feature_path(fname)] = feat
                if limit is not None and len(teacher_bank) >= int(limit):
                    break
            if limit is not None and len(teacher_bank) >= int(limit):
                break
    if was_training:
        model.train()
    return teacher_bank
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = \
        all_train_sets[set_index]

    features_all_new, labels_all, fnames_all, camids_all, \
    features_mean_new, labels_named = extract_features_voro(
        model, init_loader_new, get_mean_feature=True
    )

    features_all_old, _, _, _, features_mean_old, _ = extract_features_voro(
        model_old, init_loader_new, get_mean_feature=True
    )

    features_all_new = _subsample_feature_list(features_all_new, args.alpha_sample_limit)
    features_all_old = _subsample_feature_list(features_all_old, args.alpha_sample_limit)
    features_all_new = torch.stack(features_all_new, dim=0).cpu()
    features_all_old = torch.stack(features_all_old, dim=0).cpu()

    Affin_new = get_normal_affinity(features_all_new)
    Affin_old = get_normal_affinity(features_all_old)

    Difference = torch.abs(Affin_new - Affin_old).sum(-1).mean()

    alpha = float(torch.clamp(1 - Difference, min=0.0, max=1.0))

    return alpha


def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None,
                  stage_reference_model=None):
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]

    Epochs= args.epochs0 if 0==set_index else args.epochs
    if set_index<=1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])  # get person number in existing domains

    # =========================================================
    # =========================================================
    if set_index>0:
        '''store the old model'''
        old_model = copy.deepcopy(stage_reference_model) if stage_reference_model is not None else copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        # after sampling rehearsal, recalculate the addnum(historical ID number)
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim
        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()    
        # Initialize classifer with class centers
        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        model.cuda()

    prototype_bank = load_vprtempo_prototype_bank(args.vprtempo_distill_manifest, dataset) if args.vprtempo_distill_manifest else None
    if prototype_bank is not None:
        if prototype_bank['dim'] == out_channel:
            model.module.distill_projector = nn.Identity()
        else:
            model.module.distill_projector = nn.Linear(out_channel, prototype_bank['dim'], bias=False)
            nn.init.kaiming_normal_(model.module.distill_projector.weight, nonlinearity='linear')
        model.cuda()

    teacher_feature_bank = None
    if float(getattr(args, 'place_teacher_sim_weight', 0.0) or 0.0) > 0:
        if args.place_teacher_feature_manifest:
            teacher_feature_bank = load_teacher_feature_bank(args.place_teacher_feature_manifest, dataset)
            if teacher_feature_bank is not None:
                print('Teacher similarity bank [{}]: source=manifest, samples={}'.format(
                    name, len(teacher_feature_bank)))
        elif args.place_teacher_sim_self:
            teacher_feature_bank = build_self_teacher_feature_bank(
                model,
                init_loader,
                limit=args.place_teacher_sim_limit,
            )
            print('Teacher similarity bank [{}]: source=self-stage-start, samples={}'.format(
                name, len(teacher_feature_bank)))

    # Re-initialize optimizer
    # =========================================================
    # =========================================================
    trainable_params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        trainable_params.append(value)
    params = [{"params": trainable_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        try:
            optimizer = torch.optim.Adam(params, foreach=True)
        except TypeError:
            optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        try:
            optimizer = torch.optim.SGD(params, momentum=args.momentum, foreach=True)
        except TypeError:
            optimizer = torch.optim.SGD(params, momentum=args.momentum)
    Stones=args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    # =========================================================
    # =========================================================
  
    trainer = Trainer(cfg, args, model, add_num + num_classes,  writer=writer)
    trainer.set_external_prototypes(prototype_bank, add_num=add_num)
    trainer.set_teacher_feature_bank(teacher_feature_bank)
    raw_memory_loader = None
    if (set_index > 0 and getattr(args, 'task_type', None) == 'place' and
            getattr(args, 'MODEL', None) == 'vprtempo_snn' and
            (float(getattr(args, 'place_raw_distill_weight', 0.0) or 0.0) > 0 or
             float(getattr(args, 'place_projected_distill_weight', 0.0) or 0.0) > 0) and
            (set_index + 1) >= int(getattr(args, 'place_raw_distill_start_stage', 2) or 2)):
        if bool(getattr(args, 'place_raw_memory_all_previous', False)):
            memory_items = [(all_train_sets[i][0], all_train_sets[i][-1]) for i in range(set_index)]
            raw_memory_loader = _build_multi_raw_memory_loader(memory_items, args)
        else:
            memory_dataset, _, _, _, _, memory_name = all_train_sets[set_index - 1]
            raw_memory_loader = _build_raw_memory_loader(memory_dataset, memory_name, args)
        if raw_memory_loader is not None:
            raw_memory_loader.new_epoch()

    print('####### starting training on {} #######'.format(name))
    # =========================================================
    # =========================================================
    for epoch in range(0, Epochs):
        train_loader.new_epoch()
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model,
                      raw_memory_loader=raw_memory_loader,
                      )
        lr_scheduler.step()

        # =========================================================
        # =========================================================
        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))
            logger_res.append('epoch: {}'.format(epoch + 1))
            mAP=0.
            if args.middle_test:
                mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, args=args)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))    

    return model 

def test_model(model, all_train_sets, all_test_sets, set_index, logger_res=None, args=None):
    if args is not None and args.task_type == 'place' and args.place_eval_backend == 'vpr' and args.place_offline_eval:
        return test_model_place_offline(model, all_train_sets, all_test_sets, set_index, logger_res=logger_res, args=args)
    begin = 0
    evaluator = Evaluator(model)
    # =========================
    # =========================
    R1_all = []
    mAP_all = []
    names=''
    Results=''
    train_mAP=0
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        query = dataset.query[:args.eval_query_limit] if args is not None and args.eval_query_limit is not None else dataset.query
        gallery = dataset.gallery[:args.eval_gallery_limit] if args is not None and args.eval_gallery_limit is not None else dataset.gallery
        eval_workers = args.eval_workers if args is not None and getattr(args, 'eval_workers', None) is not None else args.workers
        eval_loader = get_test_loader(dataset, 256, 128, args.batch_size, eval_workers, testset=list(set(query) | set(gallery))) \
            if args is not None and (args.eval_query_limit is not None or args.eval_gallery_limit is not None) else test_loader
        print('Results on {}'.format(name))

        if args is not None and args.task_type == 'place' and args.place_eval_backend == 'vpr':
            train_R1, train_mAP = evaluate_vpr_dataset(
                model,
                eval_loader,
                query,
                gallery,
                dataset_name=name,
                nordland_tolerance=args.nordland_eval_tolerance,
                robotcar_tolerance=args.robotcar_eval_tolerance,
                tart_pose_threshold=args.tart_eval_pose_threshold,
                protocol=args.place_vpr_protocol,
            )
        else:
            train_R1, train_mAP = evaluator.evaluate(
                eval_loader,
                query,
                gallery,
                cmc_flag=True,
                dataset_name=name,
                external_matrix_dir=args.vprtempo_matrix_dir if args is not None else None,
                external_matrix_mode=args.vprtempo_matrix_mode if args is not None else 'replace',
                external_matrix_weight=args.vprtempo_matrix_weight if args is not None else 0.5,
                external_matrix_kind=args.vprtempo_matrix_kind if args is not None else 'distance',
                external_matrix_manifest=args.vprtempo_matrix_manifest if args is not None else None,
                task_type=args.task_type if args is not None else 'reid',
                external_embedding_manifest=args.vprtempo_embedding_manifest if args is not None else None,
                external_embedding_mode=args.vprtempo_embedding_mode if args is not None else 'replace',
                external_embedding_weight=args.vprtempo_embedding_weight if args is not None else 0.5,
                fusion_strategy=args.vprtempo_fusion_strategy if args is not None else 'linear',
                fusion_rrf_k=args.vprtempo_fusion_rrf_k if args is not None else 60,
                adaptive_alpha_temp=args.vprtempo_adaptive_alpha_temp if args is not None else 10.0,
                adaptive_confidence_mode=args.vprtempo_adaptive_confidence_mode if args is not None else 'gap',
                adaptive_alpha_min=args.vprtempo_adaptive_alpha_min if args is not None else 0.0,
                adaptive_alpha_max=args.vprtempo_adaptive_alpha_max if args is not None else 1.0,
                adaptive_gap_weight=args.vprtempo_adaptive_gap_weight if args is not None else 1.0,
                adaptive_entropy_weight=args.vprtempo_adaptive_entropy_weight if args is not None else 1.0,
                adaptive_maxsim_weight=args.vprtempo_adaptive_maxsim_weight if args is not None else 1.0,
                adaptive_entropy_softmax_temp=args.vprtempo_adaptive_entropy_softmax_temp if args is not None else 10.0,
                place_eval_mode=args.place_eval_mode if args is not None else 'legacy',
            )  # ,training_phase=i+1)
        R1_all.append(train_R1)
        mAP_all.append(train_mAP)
        names = names + name + '\t\t'
        Results=Results+'|{:.1f}/{:.1f}\t'.format(train_mAP* 100, train_R1* 100)
    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()

    # =========================
    # =========================
    R1_all = []
    mAP_all = []
    names_unseen = ''
    Results_unseen = ''
    for i in range(len(all_test_sets)):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_test_sets[i]
        query = dataset.query[:args.eval_query_limit] if args is not None and args.eval_query_limit is not None else dataset.query
        gallery = dataset.gallery[:args.eval_gallery_limit] if args is not None and args.eval_gallery_limit is not None else dataset.gallery
        eval_workers = args.eval_workers if args is not None and getattr(args, 'eval_workers', None) is not None else args.workers
        eval_loader = get_test_loader(dataset, 256, 128, args.batch_size, eval_workers, testset=list(set(query) | set(gallery))) \
            if args is not None and (args.eval_query_limit is not None or args.eval_gallery_limit is not None) else test_loader
        print('Results on {}'.format(name))
        if args is not None and args.task_type == 'place' and args.place_eval_backend == 'vpr':
            R1, mAP = evaluate_vpr_dataset(
                model,
                eval_loader,
                query,
                gallery,
                dataset_name=name,
                nordland_tolerance=args.nordland_eval_tolerance,
                robotcar_tolerance=args.robotcar_eval_tolerance,
                tart_pose_threshold=args.tart_eval_pose_threshold,
                protocol=args.place_vpr_protocol,
            )
        else:
            R1, mAP = evaluator.evaluate(
                eval_loader,
                query,
                gallery,
                cmc_flag=True,
                dataset_name=name,
                external_matrix_dir=args.vprtempo_matrix_dir if args is not None else None,
                external_matrix_mode=args.vprtempo_matrix_mode if args is not None else 'replace',
                external_matrix_weight=args.vprtempo_matrix_weight if args is not None else 0.5,
                external_matrix_kind=args.vprtempo_matrix_kind if args is not None else 'distance',
                external_matrix_manifest=args.vprtempo_matrix_manifest if args is not None else None,
                task_type=args.task_type if args is not None else 'reid',
                external_embedding_manifest=args.vprtempo_embedding_manifest if args is not None else None,
                external_embedding_mode=args.vprtempo_embedding_mode if args is not None else 'replace',
                external_embedding_weight=args.vprtempo_embedding_weight if args is not None else 0.5,
                fusion_strategy=args.vprtempo_fusion_strategy if args is not None else 'linear',
                fusion_rrf_k=args.vprtempo_fusion_rrf_k if args is not None else 60,
                adaptive_alpha_temp=args.vprtempo_adaptive_alpha_temp if args is not None else 10.0,
                adaptive_confidence_mode=args.vprtempo_adaptive_confidence_mode if args is not None else 'gap',
                adaptive_alpha_min=args.vprtempo_adaptive_alpha_min if args is not None else 0.0,
                adaptive_alpha_max=args.vprtempo_adaptive_alpha_max if args is not None else 1.0,
                adaptive_gap_weight=args.vprtempo_adaptive_gap_weight if args is not None else 1.0,
                adaptive_entropy_weight=args.vprtempo_adaptive_entropy_weight if args is not None else 1.0,
                adaptive_maxsim_weight=args.vprtempo_adaptive_maxsim_weight if args is not None else 1.0,
                adaptive_entropy_softmax_temp=args.vprtempo_adaptive_entropy_softmax_temp if args is not None else 10.0,
                place_eval_mode=args.place_eval_mode if args is not None else 'legacy',
            )
        R1_all.append(R1)
        mAP_all.append(mAP)
        names_unseen = names_unseen + name + '\t'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(mAP* 100, R1* 100)
    if len(mAP_all) > 0:
        aver_mAP_unseen = torch.tensor(mAP_all).mean()
        aver_R1_unseen = torch.tensor(R1_all).mean()
    else:
        aver_mAP_unseen = None
        aver_R1_unseen = None
    # =========================
    # =========================
    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))
    names = names + '|Average\t|'
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100)
    print(names)
    print(Results)
    '''_________________________'''
    if aver_mAP_unseen is not None:
        print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
        print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
        names_unseen = names_unseen + '|Average\t|'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_mAP_unseen* 100, aver_R1_unseen* 100)
        print(names_unseen)
        print(Results_unseen)
    else:
        print("No unseen-only datasets configured for this run.")
    # =========================
    # =========================
    if logger_res:
        logger_res.append(names)
        logger_res.append(Results)
        logger_res.append(Results.replace('|','').replace('/','\t'))
        if aver_mAP_unseen is not None:
            logger_res.append(names_unseen)
            logger_res.append(Results_unseen)
            logger_res.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return train_mAP



def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()
    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--eval-workers', type=int, default=None,
                        help="optional dataloader worker count used only during evaluation; defaults to --workers")
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")

    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x', 'snn_tiny', 'vprtempo_snn'])
    parser.add_argument('--snn-input-h', type=int, default=56, help="pooled SNN input height")
    parser.add_argument('--snn-input-w', type=int, default=56, help="pooled SNN input width")
    parser.add_argument('--snn-hidden-dim', type=int, default=1024, help="hidden dimension of the lightweight SNN backbone")
    parser.add_argument('--snn-embed-dim', type=int, default=512, help="embedding dimension of the lightweight SNN backbone")
    parser.add_argument('--snn-timesteps', type=int, default=4, help="number of recurrent spike integration steps")
    parser.add_argument('--snn-dropout', type=float, default=0.1, help="dropout applied to SNN embeddings for better generalization")
    parser.add_argument('--vprtempo-model-path', type=str, default=None,
                        help="path to a trained VPRTempo .pth model used as the place-recognition backbone")
    parser.add_argument('--vprtempo-input-h', type=int, default=56, help="VPRTempo spike input height")
    parser.add_argument('--vprtempo-input-w', type=int, default=56, help="VPRTempo spike input width")
    parser.add_argument('--vprtempo-patches', type=int, default=7, help="patch size used by VPRTempo ProcessImage")
    parser.add_argument('--vprtempo-preprocess-cache', action='store_true',
                        help="cache VPRTempo ProcessImage outputs to disk and reuse them across epochs")
    parser.add_argument('--vprtempo-preprocess-cache-dir', type=str, default=None,
                        help="directory for cached VPRTempo preprocessed tensors; defaults to <logs-dir>/vprtempo_preprocess_cache")
    parser.add_argument('--vprtempo-embed-dim', type=int, default=512, help="projected embedding dimension on top of raw VPRTempo module responses")
    parser.add_argument('--vprtempo-dropout', type=float, default=0.1, help="dropout applied after the VPRTempo projector")
    parser.add_argument('--vprtempo-freeze-mode', type=str, default='frozen', choices=['frozen', 'trainable'],
                        help="freeze the loaded VPRTempo modules or allow fine-tuning through the LSTKC losses")
    parser.add_argument('--vprtempo-stage-adapt', action='store_true',
                        help="before each continual place stage, adapt the underlying VPRTempo modules on the current stage train split")
    parser.add_argument('--vprtempo-stage-output-dir', type=str, default=str((osp.join('../logs/vprtempo_stage_adapt'))),
                        help="where to store per-stage adapted VPRTempo checkpoints")
    parser.add_argument('--vprtempo-stage-dataset-name', type=str, default='place_stageadapt',
                        help="dataset prefix used when generating temporary VPRTempo CSVs for stage adaptation")
    parser.add_argument('--vprtempo-stage-force-retrain', action='store_true',
                        help="retrain per-stage VPRTempo adapters even if a stage checkpoint already exists")
    parser.add_argument('--vprtempo-stage-epochs', type=int, default=1,
                        help="number of VPRTempo adaptation epochs run before each continual stage")
    parser.add_argument('--vprtempo-stage-filter', type=int, default=1,
                        help="frame stride used when building per-stage VPRTempo adaptation CSVs")
    parser.add_argument('--vprtempo-stage-train-layers', type=str, default='output_only',
                        choices=['all', 'output_only', 'output_plus_last_feature'],
                        help="which VPRTempo layers to adapt before the LSTKC stage starts")
    parser.add_argument('--vprtempo-stage-feature-ip-rate', type=float, default=0.15,
                        help="feature-layer intrinsic plasticity rate for stage adaptation")
    parser.add_argument('--vprtempo-stage-feature-stdp-rate', type=float, default=0.005,
                        help="feature-layer STDP rate for stage adaptation")
    parser.add_argument('--vprtempo-stage-output-ip-rate', type=float, default=0.15,
                        help="output-layer intrinsic plasticity rate for stage adaptation")
    parser.add_argument('--vprtempo-stage-output-stdp-rate', type=float, default=0.005,
                        help="output-layer STDP rate for stage adaptation")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help="enable mixed precision training on CUDA")
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help="enable cudnn benchmark for faster fixed-shape training when deterministic reproducibility is not required")
    parser.add_argument('--tf32', action='store_true',
                        help="enable TF32 matmul/cudnn acceleration on supported NVIDIA GPUs")
    parser.add_argument('--print-freq', type=int, default=200,
                        help="how many iterations between visible training log lines")
    parser.add_argument('--tb-log-freq', type=int, default=0,
                        help="how many iterations between TensorBoard scalar writes; 0 follows print frequency")
    parser.add_argument('--persistent-workers', action='store_true',
                        help="keep dataloader workers alive across epochs for faster loading")
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help="number of prefetched batches per dataloader worker when workers > 0")
    
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='../DATA/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
    parser.add_argument('--task-type', type=str, default='reid', choices=['reid', 'place'],
                        help="run the original person ReID setup or the new place recognition dataset pipeline")
    parser.add_argument('--place-train-seq', type=str, default='',
                        help="comma-separated continual order for place datasets, e.g. nordland_place,robotcar_place,tart_place")
    parser.add_argument('--place-preprocess-mode', type=str, default='auto', choices=['auto', 'reid', 'vprtempo'],
                        help="image preprocessing used for place training; auto selects VPRTempo preprocessing for the vprtempo_snn backbone")
    parser.add_argument('--place-num-instances', type=int, default=2,
                        help="instances per place identity sampled in each batch for place-recognition training")
    parser.add_argument('--place-train-positive-tolerance', type=int, default=0,
                        help="PID-neighbor tolerance used only by triplet loss for place training; CE still uses exact PID")
    parser.add_argument('--place-sampler-neighbor-tolerance', type=int, default=0,
                        help="PID-neighbor tolerance used by the place training sampler to put nearby places in the same mini-batch")
    parser.add_argument('--place-ce-weight', type=float, default=1.0,
                        help="CE loss weight used for place training")
    parser.add_argument('--place-triplet-weight', type=float, default=1.0,
                        help="Triplet loss weight used for place training")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
    parser.add_argument('--vprtempo-matrix-dir', type=str, default=None,
                        help="directory containing external matrices named <dataset>_qg.npy/.npz, optionally <dataset>_qq and <dataset>_gg")
    parser.add_argument('--vprtempo-matrix-manifest', type=str, default=None,
                        help="JSON manifest mapping each dataset name to qg/qq/gg matrix paths exported by VPRTempo")
    parser.add_argument('--vprtempo-embedding-manifest', type=str, default=None,
                        help="JSON manifest mapping each dataset name to external query/gallery embedding paths exported by VPRTempo")
    parser.add_argument('--vprtempo-distill-manifest', type=str, default=None,
                        help="JSON manifest mapping each dataset name to external query/gallery embedding paths used to build prototype distillation targets")
    parser.add_argument('--vprtempo-distill-weight', type=float, default=0.0,
                        help="weight for prototype distillation from VPRTempo embeddings during training")
    parser.add_argument('--vprtempo-raw-triplet-weight', type=float, default=0.0,
                        help="extra triplet loss weight applied directly to raw VPRTempo responses during place training")
    parser.add_argument('--vprtempo-raw-infonce-weight', type=float, default=0.0,
                        help="supervised contrastive InfoNCE loss weight applied directly to raw VPRTempo responses")
    parser.add_argument('--vprtempo-raw-infonce-temp', type=float, default=0.07,
                        help="temperature for raw supervised contrastive InfoNCE loss")
    parser.add_argument('--place-teacher-feature-manifest', type=str, default=None,
                        help="JSON manifest mapping dataset names to teacher query/gallery embeddings for batch similarity distillation")
    parser.add_argument('--place-teacher-sim-weight', type=float, default=0.0,
                        help="weight for batchwise teacher similarity distillation on single-frame descriptors")
    parser.add_argument('--place-teacher-sim-temp', type=float, default=0.07,
                        help="temperature for teacher/student batch similarity distributions")
    parser.add_argument('--place-teacher-sim-min-batch', type=int, default=4,
                        help="minimum number of teacher-covered samples in a batch before applying teacher similarity distillation")
    parser.add_argument('--place-teacher-sim-self', action='store_true',
                        help="use the stage-start model raw descriptors as an in-run teacher bank when no teacher manifest is provided")
    parser.add_argument('--place-teacher-sim-limit', type=int, default=0,
                        help="optional cap for self teacher samples; 0 means all init-loader samples")
    parser.add_argument('--place-raw-distill-weight', type=float, default=0.0,
                        help="cosine distillation weight that keeps raw VPRTempo responses on previous-stage memory samples")
    parser.add_argument('--place-raw-memory-size', type=int, default=1024,
                        help="number of previous-stage train samples used for raw memory distillation")
    parser.add_argument('--place-raw-memory-batch-size', type=int, default=128,
                        help="batch size for previous-stage raw memory distillation")
    parser.add_argument('--place-raw-memory-workers', type=int, default=-1,
                        help="dataloader workers for raw memory distillation; -1 follows eval-workers/workers")
    parser.add_argument('--place-raw-distill-freq', type=int, default=1,
                        help="apply raw memory distillation every N training iterations")
    parser.add_argument('--place-raw-memory-all-previous', action='store_true',
                        help="sample raw distillation memory from all previous place stages instead of only the immediate previous stage")
    parser.add_argument('--place-raw-distill-start-stage', type=int, default=2,
                        help="1-based place stage index where raw memory distillation starts")
    parser.add_argument('--place-projected-distill-weight', type=float, default=0.0,
                        help="cosine distillation weight that keeps trainable projected descriptors on previous-stage memory samples")
    parser.add_argument('--place-projected-distill-freq', type=int, default=1,
                        help="apply projected descriptor memory distillation every N training iterations")
    parser.add_argument('--vprtempo-embedding-mode', type=str, default='replace', choices=['replace', 'fuse'],
                        help="replace the model distance matrix with external embeddings or fuse model features with them")
    parser.add_argument('--vprtempo-embedding-weight', type=float, default=0.5,
                        help="weight for the model distance matrix when embedding mode is fuse")
    parser.add_argument('--vprtempo-fusion-strategy', type=str, default='linear',
                        choices=['linear', 'rowwise_linear', 'rrf', 'confidence_adaptive'],
                        help="distance fusion strategy used when combining model features with external embeddings/matrices")
    parser.add_argument('--vprtempo-fusion-rrf-k', type=int, default=60,
                        help="RRF constant used when --vprtempo-fusion-strategy=rrf")
    parser.add_argument('--vprtempo-adaptive-alpha-temp', type=float, default=10.0,
                        help="temperature used by confidence-based adaptive fusion; higher values produce sharper per-query alpha switching")
    parser.add_argument('--vprtempo-adaptive-confidence-mode', type=str, default='gap',
                        choices=['gap', 'gap_entropy', 'gap_entropy_max'],
                        help="confidence formula used by confidence-based adaptive fusion")
    parser.add_argument('--vprtempo-adaptive-alpha-min', type=float, default=0.0,
                        help="lower bound for adaptive alpha after sigmoid rescaling")
    parser.add_argument('--vprtempo-adaptive-alpha-max', type=float, default=1.0,
                        help="upper bound for adaptive alpha after sigmoid rescaling")
    parser.add_argument('--vprtempo-adaptive-gap-weight', type=float, default=1.0,
                        help="weight of the top1-top2 gap term in adaptive confidence")
    parser.add_argument('--vprtempo-adaptive-entropy-weight', type=float, default=1.0,
                        help="weight of the entropy-certainty term in adaptive confidence")
    parser.add_argument('--vprtempo-adaptive-maxsim-weight', type=float, default=1.0,
                        help="weight of the max-similarity term in adaptive confidence")
    parser.add_argument('--vprtempo-adaptive-entropy-softmax-temp', type=float, default=10.0,
                        help="softmax temperature used when converting similarities into entropy-based confidence")
    parser.add_argument('--vprtempo-matrix-mode', type=str, default='replace', choices=['replace', 'fuse'],
                        help="replace the model distance matrix or fuse it with the external matrix")
    parser.add_argument('--vprtempo-matrix-weight', type=float, default=0.5,
                        help="weight for the model distance matrix when matrix mode is fuse")
    parser.add_argument('--vprtempo-matrix-kind', type=str, default='distance', choices=['distance', 'similarity'],
                        help="whether the external matrix stores distances or similarities")
    parser.add_argument('--eval-query-limit', type=int, default=None,
                        help="optional cap on the number of query samples used during evaluation")
    parser.add_argument('--eval-gallery-limit', type=int, default=None,
                        help="optional cap on the number of gallery samples used during evaluation")
    parser.add_argument('--place-eval-mode', type=str, default='legacy', choices=['legacy', 'blockwise'],
                        help="place evaluation implementation; use blockwise only after validating against legacy")
    parser.add_argument('--place-eval-backend', type=str, default='vpr', choices=['reid', 'vpr'],
                        help="evaluation backend for place tasks; vpr is recommended and now the default")
    parser.add_argument('--place-offline-eval', action='store_true',
                        help="for place+vpr evaluation, export query/gallery embeddings once and run offline VPR ranking from embeddings")
    parser.add_argument('--place-offline-eval-dir', type=str, default=None,
                        help="directory used to store exported query/gallery embeddings and the offline evaluation manifest")
    parser.add_argument('--place-offline-eval-feature', type=str, default='projected',
                        choices=['projected', 'raw', 'bn', 'auto'],
                        help="feature kind used by offline place evaluation; auto exports projected+raw for vprtempo_snn")
    parser.add_argument('--place-offline-eval-primary-feature', type=str, default='first',
                        choices=['first', 'last'],
                        help="which feature result is written to log_res when multiple offline features are evaluated")
    parser.add_argument('--place-sequence-rerank', action='store_true',
                        help="apply VPR sequence-consistency reranking during place offline evaluation")
    parser.add_argument('--place-sequence-radius', type=int, default=30,
                        help="temporal/frame radius used by sequence-consistency reranking")
    parser.add_argument('--place-sequence-weight', type=float, default=1.0,
                        help="weight of sequence-consistency score during reranking")
    parser.add_argument('--place-sequence-candidate-topk', type=int, default=3000,
                        help="base retrieval candidate count reranked by sequence-consistency")
    parser.add_argument('--place-sequence-query-block-size', type=int, default=16,
                        help="query block size for streaming sequence reranking")
    parser.add_argument('--place-sequence-candidate-chunk-size', type=int, default=64,
                        help="candidate chunk size for streaming sequence reranking")
    parser.add_argument('--place-vpr-protocol', type=str, default='dataset', choices=['dataset', 'strict_pid'],
                        help="positive-match protocol for the VPR evaluator")
    parser.add_argument('--nordland-eval-tolerance', type=int, default=0,
                        help="frame tolerance used by the VPR evaluator on nordland_place")
    parser.add_argument('--robotcar-eval-tolerance', type=int, default=1,
                        help="frame tolerance used by the VPR evaluator on robotcar_place")
    parser.add_argument('--tart-eval-pose-threshold', type=float, default=3.0,
                        help="xy pose threshold used by the VPR evaluator on tart_place")
    parser.add_argument('--place-train-limit', type=int, default=None,
                        help="optional cap on the number of place IDs used for fast sequential-training experiments")
    parser.add_argument('--alpha-sample-limit', type=int, default=1024,
                        help="optional cap on feature samples used when estimating adaptive alpha")
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--max-stages', type=int, default=None, help="limit the number of training stages for debugging")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")    
    parser.add_argument('--disable-stage-model-fusion', action='store_true',
                        help="disable adaptive parameter fusion after each place stage")
    main()

