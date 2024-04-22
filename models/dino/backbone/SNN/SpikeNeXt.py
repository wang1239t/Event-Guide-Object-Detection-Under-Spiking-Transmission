# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections import OrderedDict

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.dino.backbone.SNN.layers import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from util.misc import NestedTensor


# from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, steps=5, norm_layer=None):
        super().__init__()
        self.s_dwconv = tdLayer(nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
                                norm_layer(dim, eps=1e-6))  # depthwise conv

        self.s_pwconv1 = tdLayer(nn.Conv2d(dim, 4 * dim, kernel_size=1, stride=1,
                                           bias=False))  # pointwise/1x1 convs, implemented with linear
        # layers
        self.act = LIFSpike()
        self.s_pwconv2 = tdLayer(nn.Conv2d(4 * dim, dim, kernel_size=1, stride=1, bias=False))
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, steps)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.act(x)
        x = self.s_dwconv(x)
        x = self.act(x)
        x = self.s_pwconv1(x)
        x = self.act(x)
        x = self.s_pwconv2(x)
        x = x.permute(0, 2, 3, 1, 4)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2, 4)
        x = input + self.drop_path(x)
        return x


class SpikeNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 out_indices=[0, 1, 2, 3], norm_layer=None, steps=5
                 ):
        super().__init__()
        self.steps = steps
        self.dims = dims
        if norm_layer is None:
            self._norm_layer = tdBatchNorm
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            tdLayer(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), tdBatchNorm(dims[0], eps=1e-6))
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                tdBatchNorm(dims[i], eps=1e-6),
                LIFSpike(),
                tdLayer(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, norm_layer=self._norm_layer, steps=self.steps)
                  for j
                  in
                  range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layers = partial(tdBatchNorm, eps=1e-6)
        for i_layer in range(4):
            layer = norm_layers(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        outs = OrderedDict()
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layers = getattr(self, f'norm{i}')
                x_out = norm_layers(x)
                outs[i] = torch.sum(x_out, dim=4) / self.steps
        # return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return outs

    def forward(self, x):
        out = self.forward_features(x)
        ps = []
        return out, ps

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if int(name) in output_layers:
            outputs[name] = torch.sum(x, dim=4) / self.steps
        return int(name) == output_layers[-1]


def spikenext(modelname, steps, **kw):
    assert modelname in ['spikenext-T', 'spikenext-B', 'spikenext-L']

    model_para_dict = {
        'spikenext-B': dict(
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
        ),
        'spikenext-L': dict(
            depths=[3, 3, 27, 3],
            dims=[96, 192, 384, 768],
        ),
        'spikenext-T': dict(
            depths=[2, 2, 8, 2],
            dims=[96, 192, 384, 768],
        ),
    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    model = SpikeNeXt(**kw_cgf, steps=steps)
    dims = model_para_dict[modelname]['dims']

    return model, dims

