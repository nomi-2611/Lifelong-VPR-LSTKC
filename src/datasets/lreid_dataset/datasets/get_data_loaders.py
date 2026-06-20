import copy
import os.path
import os
import sys
import hashlib
from pathlib import Path

import torch
from src.utils.reid_utils.feature_tools import *
import src.datasets.lreid_dataset.datasets as datasets
from src.utils.reid_utils.data.sampler import PlaceNeighborIdentitySampler, RandomIdentitySampler, RandomMultipleGallerySampler
from src.utils.reid_utils.data import IterLoader
import numpy as np

_VPRTEMPO_ROOT = Path(__file__).resolve().parents[3] / "feature_extraction" / "vprtempo_snn"
if _VPRTEMPO_ROOT.exists() and str(_VPRTEMPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_VPRTEMPO_ROOT))

try:
    from vprtempo.src.dataset import ProcessImage as VPRTempoProcessImage
except Exception:
    VPRTempoProcessImage = None


class VPRTempoPlaceTransform(object):
    def __init__(self, dims=(56, 56), patches=7):
        if VPRTempoProcessImage is None:
            raise ImportError("VPRTempo ProcessImage is unavailable. Check the integrated src/feature_extraction/vprtempo_snn module.")
        self.processor = VPRTempoProcessImage(dims, patches)
        self.to_tensor = T.PILToTensor() if hasattr(T, "PILToTensor") else None

    def __call__(self, img):
        if torch.is_tensor(img):
            tensor = img.contiguous()
        elif self.to_tensor is not None:
            tensor = self.to_tensor(img)
        else:
            array = np.asarray(img)
            if array.ndim == 2:
                array = array[:, :, None]
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return self.processor(tensor)


def _resolve_place_preprocess_mode(cfg, dataset_name):
    if not dataset_name.endswith('_place'):
        return 'reid'
    mode = getattr(cfg, 'place_preprocess_mode', 'reid')
    if mode == 'auto':
        return 'vprtempo' if getattr(cfg, 'MODEL', None) == 'vprtempo_snn' else 'reid'
    return mode


def _build_transforms(cfg, dataset_name, height, width):
    if _resolve_place_preprocess_mode(cfg, dataset_name) == 'vprtempo':
        dims = (
            int(getattr(cfg, 'vprtempo_input_h', 56)),
            int(getattr(cfg, 'vprtempo_input_w', 56)),
        )
        patches = int(getattr(cfg, 'vprtempo_patches', 7))
        place_transform = VPRTempoPlaceTransform(dims=dims, patches=patches)
        return place_transform, place_transform

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    return train_transformer, test_transformer


def _current_cfg_proxy():
    return type("CfgProxy", (), {
        "place_preprocess_mode": getattr(get_data, "_place_preprocess_mode", "reid"),
        "MODEL": getattr(get_data, "_model_name", "50x"),
        "vprtempo_input_h": getattr(get_data, "_vprtempo_input_h", 56),
        "vprtempo_input_w": getattr(get_data, "_vprtempo_input_w", 56),
        "vprtempo_patches": getattr(get_data, "_vprtempo_patches", 7),
    })()


def _dataloader_kwargs(workers):
    kwargs = {"pin_memory": True}
    if int(workers) > 0:
        kwargs["persistent_workers"] = bool(getattr(get_data, "_persistent_workers", False))
        prefetch_factor = int(getattr(get_data, "_prefetch_factor", 2))
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _place_cache_enabled(name):
    return bool(getattr(get_data, "_vprtempo_preprocess_cache", False)) and _resolve_place_preprocess_mode(
        _current_cfg_proxy(), name) == 'vprtempo'


def _build_cache_namespace(name, split_tag):
    payload = 'dataset={}|split={}|mode={}|model={}|h={}|w={}|patches={}'.format(
        name,
        split_tag,
        getattr(get_data, "_place_preprocess_mode", "reid"),
        getattr(get_data, "_model_name", "50x"),
        getattr(get_data, "_vprtempo_input_h", 56),
        getattr(get_data, "_vprtempo_input_w", 56),
        getattr(get_data, "_vprtempo_patches", 7),
    )
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _build_preprocessor(dataset_entries, dataset_root, transform, name, split_tag):
    cache_root = None
    cache_namespace = None
    if _place_cache_enabled(name):
        cache_root = getattr(get_data, "_vprtempo_cache_dir", None)
        cache_namespace = _build_cache_namespace(name, split_tag)
    return Preprocessor(
        dataset_entries,
        root=dataset_root,
        transform=transform,
        cache_root=cache_root,
        cache_namespace=cache_namespace,
    )

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, select_num=0):
    # root = osp.join(data_dir, name)
    root = data_dir
    
    dataset = datasets.create(name, root)
    '''select some persons for training'''
    if select_num > 0:
        train = []
        for instance in dataset.train:
            if instance[1] < select_num:
                train.append((instance[0], instance[1], instance[2], instance[3]))

        dataset.train = train
        dataset.num_train_pids = len({instance[1] for instance in train})
        dataset.num_train_imgs = len(train)

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer, test_transformer = _build_transforms(
        cfg=_current_cfg_proxy(),
        dataset_name=name,
        height=height,
        width=width,
    )

    sampler_instances = num_instances
    if name.endswith('_place'):
        sampler_instances = max(1, int(getattr(get_data, "_place_num_instances", 2)))
    rmgs_flag = sampler_instances > 0
    if rmgs_flag:
        if name.endswith('_place'):
            place_neighbor_tolerance = int(getattr(get_data, "_place_sampler_neighbor_tolerance", 0) or 0)
            if place_neighbor_tolerance > 0:
                sampler = PlaceNeighborIdentitySampler(
                    train_set,
                    sampler_instances,
                    neighbor_tolerance=place_neighbor_tolerance,
                )
            else:
                sampler = RandomIdentitySampler(train_set, sampler_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, sampler_instances)
    else:
        sampler = None


    train_loader = IterLoader(
        DataLoader(_build_preprocessor(train_set, dataset.images_dir, train_transformer, name, 'train'),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, drop_last=True, **_dataloader_kwargs(workers)), length=iters)

    test_loader = DataLoader(
        _build_preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     dataset.images_dir, test_transformer, name, 'test'),
        batch_size=batch_size, num_workers=workers, shuffle=False, **_dataloader_kwargs(workers))

    init_loader = DataLoader(_build_preprocessor(train_set, dataset.images_dir, test_transformer, name, 'init'),
                             batch_size=128, num_workers=workers, shuffle=False, drop_last=False,
                             **_dataloader_kwargs(workers))

    return [dataset, num_classes, train_loader, test_loader, init_loader, name]

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    _, test_transformer = _build_transforms(
        cfg=_current_cfg_proxy(),
        dataset_name=getattr(dataset, 'dataset_name', ''),
        height=height,
        width=width,
    )

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        _build_preprocessor(testset, dataset.images_dir, test_transformer,
                            getattr(dataset, 'dataset_name', ''), 'test'),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, **_dataloader_kwargs(workers))

    return test_loader


def build_data_loaders(cfg, training_set, testing_only_set, toy_num=0):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = (256, 128)
    get_data._place_preprocess_mode = getattr(cfg, 'place_preprocess_mode', 'reid')
    get_data._model_name = getattr(cfg, 'MODEL', '50x')
    get_data._vprtempo_input_h = getattr(cfg, 'vprtempo_input_h', 56)
    get_data._vprtempo_input_w = getattr(cfg, 'vprtempo_input_w', 56)
    get_data._vprtempo_patches = getattr(cfg, 'vprtempo_patches', 7)
    get_data._place_num_instances = getattr(cfg, 'place_num_instances', 2)
    get_data._place_sampler_neighbor_tolerance = getattr(cfg, 'place_sampler_neighbor_tolerance', 0)
    get_data._persistent_workers = getattr(cfg, 'persistent_workers', False)
    get_data._prefetch_factor = getattr(cfg, 'prefetch_factor', 2)
    get_data._vprtempo_preprocess_cache = getattr(cfg, 'vprtempo_preprocess_cache', False)
    get_data._vprtempo_cache_dir = getattr(cfg, 'vprtempo_preprocess_cache_dir', None)
    place_select_num = getattr(cfg, 'place_train_limit', None)
    def _select_num_for_training(name):
        if name.endswith('_place'):
            return int(place_select_num) if place_select_num is not None else 0
        return 500

    training_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances,
                                 select_num=_select_num_for_training(name)) for name in training_set]

  
    testing_loaders = [get_data(name, data_dir, height, width, cfg.batch_size, cfg.workers,
                                cfg.num_instances) for name in testing_only_set]
    return training_loaders, testing_loaders

