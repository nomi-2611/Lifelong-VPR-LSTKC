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
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.datasets.lreid_dataset.datasets as datasets


class ImagePathDataset(Dataset):
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path = self.entries[index][0]
        image = Image.open(path).convert('RGB')
        return self.transform(image), path


class ResNet50Teacher(nn.Module):
    def __init__(self):
        super(ResNet50Teacher, self).__init__()
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


class TorchvisionTeacher(nn.Module):
    def __init__(self, model_name):
        super(TorchvisionTeacher, self).__init__()
        if model_name == 'vit_b_16':
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights)
            model.heads = nn.Identity()
        elif model_name == 'convnext_small':
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            model = models.convnext_small(weights=weights)
            model.classifier[-1] = nn.Identity()
        elif model_name == 'convnext_base':
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            model = models.convnext_base(weights=weights)
            model.classifier[-1] = nn.Identity()
        elif model_name == 'swin_t':
            weights = models.Swin_T_Weights.IMAGENET1K_V1
            model = models.swin_t(weights=weights)
            model.head = nn.Identity()
        else:
            raise ValueError('Unsupported torchvision teacher: {}'.format(model_name))
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


def parse_args():
    parser = argparse.ArgumentParser(description='Export external single-frame teacher features for place datasets.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dataset-names', nargs='+', default=['robotcar_place', 'nordland_place', 'pitts30k_place'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--teacher',
                        choices=['dinov2_vits14', 'resnet50', 'vit_b_16', 'convnext_small', 'convnext_base', 'swin_t'],
                        default='dinov2_vits14')
    parser.add_argument('--splits', nargs='+', default=['train', 'query', 'gallery'],
                        choices=['train', 'query', 'gallery'])
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--train-limit', type=int, default=None)
    parser.add_argument('--query-limit', type=int, default=None)
    parser.add_argument('--gallery-limit', type=int, default=None)
    parser.add_argument('--fallback-resnet50', action='store_true',
                        help='fallback to torchvision ResNet50 if DINOv2 cannot be loaded')
    return parser.parse_args()


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_teacher(name, fallback_resnet50=False):
    if name == 'resnet50':
        return ResNet50Teacher(), 'resnet50'
    if name in ('vit_b_16', 'convnext_small', 'convnext_base', 'swin_t'):
        return TorchvisionTeacher(name), name
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        return model, 'dinov2_vits14'
    except Exception as exc:
        if not fallback_resnet50:
            raise
        print('DINOv2 load failed: {}'.format(exc))
        print('Falling back to torchvision ResNet50 teacher.')
        return ResNet50Teacher(), 'resnet50'


def extract_embeddings(model, entries, transform, batch_size, workers):
    if not entries:
        return np.zeros((0, 0), dtype=np.float32)
    loader = DataLoader(
        ImagePathDataset(entries, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    outputs_all = []
    start = time.time()
    model.eval()
    with torch.no_grad():
        for index, (images, paths) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            outputs = torch.nn.functional.normalize(outputs.float(), dim=1)
            outputs_all.append(outputs.cpu().numpy().astype(np.float32))
            if (index + 1) % 20 == 0 or (index + 1) == len(loader):
                print('  batches {}/{} elapsed {:.1f}s'.format(index + 1, len(loader), time.time() - start))
    return np.concatenate(outputs_all, axis=0).astype(np.float32)


def _limit(entries, limit):
    if limit is None:
        return entries
    return entries[:int(limit)]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    transform = build_transform(args.image_size)
    teacher, teacher_name = build_teacher(args.teacher, fallback_resnet50=args.fallback_resnet50)
    teacher.cuda()

    manifest = OrderedDict()
    summary = OrderedDict()

    for dataset_name in args.dataset_names:
        dataset = datasets.create(dataset_name, args.data_dir)
        split_entries = {
            'train': _limit(dataset.train, args.train_limit),
            'query': _limit(dataset.query, args.query_limit),
            'gallery': _limit(dataset.gallery, args.gallery_limit),
        }
        manifest_item = {
            'teacher': teacher_name,
            'metric': 'cosine',
            'normalized': True,
            'image_size': int(args.image_size),
        }
        summary[dataset_name] = {}
        for split in args.splits:
            entries = split_entries[split]
            print('Exporting {} {} teacher={} samples={}'.format(dataset_name, split, teacher_name, len(entries)))
            embeddings = extract_embeddings(teacher, entries, transform, args.batch_size, args.workers)
            out_path = osp.abspath(osp.join(args.output_dir, '{}_{}_embeddings.npy'.format(dataset_name, split))).replace('\\', '/')
            np.save(out_path, embeddings)
            manifest_item['{}_embeddings'.format(split)] = out_path
            summary[dataset_name][split] = {
                'count': int(embeddings.shape[0]),
                'dim': int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else 0,
            }
        manifest[dataset_name] = manifest_item

    manifest_path = osp.join(args.output_dir, 'teacher_embedding_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    summary_path = osp.join(args.output_dir, 'teacher_embedding_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({
        'manifest_path': osp.abspath(manifest_path).replace('\\', '/'),
        'summary_path': osp.abspath(summary_path).replace('\\', '/'),
        'teacher': teacher_name,
    }, indent=2))


if __name__ == '__main__':
    main()
