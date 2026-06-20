from __future__ import absolute_import

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F


VPRTEMPO_ROOT = Path(__file__).resolve().parents[2] / "feature_extraction" / "vprtempo_snn"
if VPRTEMPO_ROOT.exists() and str(VPRTEMPO_ROOT) not in sys.path:
    sys.path.insert(0, str(VPRTEMPO_ROOT))


def _load_vprtempo_state(model_path):
    if model_path is None:
        raise ValueError("vprtempo_snn requires --vprtempo-model-path")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("VPRTempo model not found: {}".format(model_path))
    try:
        return torch.load(str(model_path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(model_path), map_location="cpu")


def _sorted_model_keys(state):
    return sorted(state.keys(), key=lambda key: int(key.split("_")[1]))


class VPRTempoBackbone(nn.Module):
    def __init__(self, num_classes, model_path, input_hw=(56, 56), patches=7,
                 embed_dim=512, dropout=0.1, freeze_mode='frozen'):
        super(VPRTempoBackbone, self).__init__()
        state = _load_vprtempo_state(model_path)
        model_keys = _sorted_model_keys(state)
        if not model_keys:
            raise ValueError("No VPRTempo modules found in {}".format(model_path))

        modules = []
        raw_dim = 0
        input_dim = int(input_hw[0] * input_hw[1])
        for key in model_keys:
            module_state = state[key]
            feature_weight = module_state['feature_layer.w.weight']
            output_weight = module_state['output_layer.w.weight']
            feature_layer = nn.Linear(input_dim, feature_weight.shape[0], bias=False)
            output_layer = nn.Linear(feature_weight.shape[0], output_weight.shape[0], bias=False)
            feature_layer.weight.data.copy_(feature_weight.float())
            output_layer.weight.data.copy_(output_weight.float())
            modules.append(nn.Sequential(feature_layer, output_layer))
            raw_dim += int(output_weight.shape[0])

        self.base = nn.ModuleList(modules)
        self.in_planes = embed_dim
        self.freeze_mode = freeze_mode
        self.model_path = str(model_path)
        self.input_hw = input_hw
        self.patches = patches
        self.raw_dim = raw_dim
        self.projector = nn.Linear(raw_dim, embed_dim, bias=False)
        nn.init.kaiming_normal_(self.projector.weight, nonlinearity='linear')
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        if freeze_mode == 'frozen':
            for param in self.base.parameters():
                param.requires_grad_(False)
        print('using vprtempo_snn as a backbone (raw_dim={}, embed_dim={}, freeze_mode={})'.format(
            raw_dim, embed_dim, freeze_mode
        ))

    def _set_freeze_mode(self, freeze_mode):
        self.freeze_mode = freeze_mode
        requires_grad = freeze_mode != 'frozen'
        for param in self.base.parameters():
            param.requires_grad_(requires_grad)

    def reload_base_from_model_path(self, model_path, freeze_mode=None):
        state = _load_vprtempo_state(model_path)
        model_keys = _sorted_model_keys(state)
        if len(model_keys) != len(self.base):
            raise ValueError('Reloaded VPRTempo module count does not match the current backbone')
        raw_dim = 0
        for module, key in zip(self.base, model_keys):
            module_state = state[key]
            feature_weight = module_state['feature_layer.w.weight'].float()
            output_weight = module_state['output_layer.w.weight'].float()
            if module[0].weight.shape != feature_weight.shape:
                raise ValueError('Feature layer shape mismatch during VPRTempo reload')
            if module[1].weight.shape != output_weight.shape:
                raise ValueError('Output layer shape mismatch during VPRTempo reload')
            module[0].weight.data.copy_(feature_weight)
            module[1].weight.data.copy_(output_weight)
            raw_dim += int(output_weight.shape[0])
        if raw_dim != self.raw_dim:
            raise ValueError('Reloaded VPRTempo raw dimension mismatch')
        self.model_path = str(model_path)
        self._set_freeze_mode(self.freeze_mode if freeze_mode is None else freeze_mode)

    def _compute_raw_feat(self, x):
        with torch.set_grad_enabled(self.freeze_mode != 'frozen' and self.training):
            outputs = [module(x) for module in self.base]
            raw_feat = torch.cat(outputs, dim=1)
        if self.freeze_mode == 'frozen':
            raw_feat = raw_feat.detach()
        return raw_feat

    def forward_from_raw(self, raw_feat, get_all_feat=False):
        projected = self.projector(raw_feat)
        projected = self.dropout(projected)
        global_feat = F.normalize(projected, dim=1)
        bn_feat = self.bottleneck(global_feat)
        feat_map = global_feat.unsqueeze(-1).unsqueeze(-1)

        if get_all_feat is True:
            cls_outputs = self.classifier(bn_feat)
            return global_feat, bn_feat, cls_outputs, feat_map

        if self.training is False:
            return global_feat

        cls_outputs = self.classifier(bn_feat)
        return global_feat, bn_feat, cls_outputs, feat_map

    def forward_with_raw(self, x, domains=None, training_phase=None, get_all_feat=False, epoch=0):
        raw_feat = self._compute_raw_feat(x)
        outputs = self.forward_from_raw(raw_feat, get_all_feat=get_all_feat)
        if get_all_feat is True:
            return outputs[0], outputs[1], outputs[2], outputs[3], raw_feat
        if self.training is False:
            return outputs, raw_feat
        return outputs[0], outputs[1], outputs[2], outputs[3], raw_feat

    def extract_raw_feature(self, x):
        return self._compute_raw_feat(x)

    def forward(self, x, domains=None, training_phase=None, get_all_feat=False, epoch=0):
        outputs = self.forward_with_raw(
            x,
            domains=domains,
            training_phase=training_phase,
            get_all_feat=get_all_feat,
            epoch=epoch,
        )
        if get_all_feat is True:
            return outputs[0], outputs[1], outputs[2], outputs[3]
        if self.training is False:
            return outputs[0]
        return outputs[0], outputs[1], outputs[2], outputs[3]


