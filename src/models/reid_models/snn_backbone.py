import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        grad = torch.clamp(1.0 - input_tensor.abs(), min=0.0)
        return grad_output * grad


class SpikeEncoder(nn.Module):
    def __init__(self, input_hw):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(input_hw)

    def forward(self, x):
        # Use grayscale pooled frames for a lighter, more robust SNN front-end.
        x = x.mean(dim=1, keepdim=True)
        x = self.pool(x)
        return x.flatten(1)


class SpikeLinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, threshold=1.0, decay=0.5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.threshold = threshold
        self.decay = decay
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="linear")

    def forward(self, x, timesteps):
        mem = torch.zeros(x.size(0), self.linear.out_features, device=x.device, dtype=x.dtype)
        spike_sum = torch.zeros_like(mem)
        for _ in range(timesteps):
            cur = self.norm(self.linear(x))
            mem = mem * self.decay + cur
            spikes = SurrogateSpike.apply(mem - self.threshold)
            spike_sum = spike_sum + spikes
            mem = mem - spikes * self.threshold
        return spike_sum / float(timesteps)


class IdentityBase(nn.Module):
    def __init__(self, encoder, hidden_block, output_block):
        super().__init__()
        self.encoder = encoder
        self.hidden_block = hidden_block
        self.output_block = output_block

    def forward(self, x, timesteps):
        x = self.encoder(x)
        x = self.hidden_block(x, timesteps)
        x = self.output_block(x, timesteps)
        return x


class SNNBackbone(nn.Module):
    def __init__(self, num_classes, input_hw=(56, 56), hidden_dim=1024, embed_dim=512, timesteps=4, dropout=0.1):
        super().__init__()
        self.in_planes = embed_dim
        input_dim = input_hw[0] * input_hw[1]
        self.base = IdentityBase(
            SpikeEncoder(input_hw),
            SpikeLinearBlock(input_dim, hidden_dim),
            SpikeLinearBlock(hidden_dim, embed_dim),
        )
        self.timesteps = timesteps
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        print(f'using snn_tiny as a backbone (input={input_hw[0]}x{input_hw[1]}, hidden={hidden_dim}, embed={embed_dim}, T={timesteps})')

    def forward(self, x, domains=None, training_phase=None, get_all_feat=False, epoch=0):
        feat = self.base(x, self.timesteps)
        feat = self.dropout(feat)
        global_feat = F.normalize(feat, dim=1)
        bn_feat = self.bottleneck(global_feat)
        feat_map = global_feat.unsqueeze(-1).unsqueeze(-1)

        if get_all_feat is True:
            cls_outputs = self.classifier(bn_feat)
            return global_feat, bn_feat, cls_outputs, feat_map

        if self.training is False:
            return global_feat

        cls_outputs = self.classifier(bn_feat)
        return global_feat, bn_feat, cls_outputs, feat_map
