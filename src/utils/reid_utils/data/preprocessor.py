from __future__ import absolute_import
import os
import os.path as osp
import hashlib
import time
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, cache_root=None, cache_namespace=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.cache_root = cache_root
        self.cache_namespace = cache_namespace

    def _resolve_path(self, fname):
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        return fpath

    def _cache_path(self, fpath):
        if not self.cache_root or not self.cache_namespace:
            return None
        digest = hashlib.sha1(osp.normpath(str(fpath)).encode('utf-8', errors='replace')).hexdigest()
        return osp.join(self.cache_root, self.cache_namespace, digest[:2], digest + '.pt')

    def _load_or_build_cached_tensor(self, fpath):
        cache_path = self._cache_path(fpath)
        if cache_path is not None and osp.exists(cache_path):
            for _ in range(10):
                try:
                    return torch.load(cache_path, map_location='cpu')
                except PermissionError:
                    time.sleep(0.05)

        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if cache_path is not None and torch.is_tensor(img):
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            temp_path = cache_path + '.tmp-{}'.format(os.getpid())
            try:
                torch.save(img.cpu(), temp_path)
                os.replace(temp_path, cache_path)
            except Exception:
                if osp.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        # print(self.dataset)
        try:
            fname, pid, camid, domain = self.dataset[index]
        except:
            fname, pid, camid, domain, _ = self.dataset[index]
        fpath = self._resolve_path(fname)
        img = self._load_or_build_cached_tensor(fpath)

        return img, fname, pid, camid, domain
