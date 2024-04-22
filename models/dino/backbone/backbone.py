# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from typing import Dict, List

from models.dino.backbone.SNN.EMS_ResNet import resnet50_ems_snn
from models.dino.backbone.SNN.SEW_ResNet import sew_resnet50_snn
from models.dino.backbone.SNN.Vanilla_ResNet import resnet50_snn
from models.dino.backbone.SNN.MS_ResNet import ms_resnet50_snn
from util.misc import NestedTensor

from models.dino.backbone.position_encoding import build_position_encoding
from models.dino.backbone.SNN.SpikeNeXt import spikenext


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels, name=None):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels
        self.name = name

    def forward(self, tensor_list: NestedTensor):
        B_size, N_steps, H, W = tensor_list.tensors.shape
        tensor_list.tensors = tensor_list.tensors.view(B_size, -1, 3, H, W).permute(0, 2, 3, 4, 1)
        xs, fg_pos = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_indices: list, steps: int):
        if name == 'resnet50-sew-snn':
            backbone = sew_resnet50_snn(return_interm_indices=return_interm_indices, steps=steps)
            num_channels_all = [256, 512, 1024, 2048]
        elif name == 'resnet50-ems-snn':
            backbone = resnet50_ems_snn(return_interm_indices=return_interm_indices, steps=steps)
            num_channels_all = [256, 512, 1024, 2048]
        elif name == 'resnet50-snn':
            backbone = resnet50_snn(return_interm_indices=return_interm_indices, steps=steps)
            num_channels_all = [256, 512, 1024, 2048]
        elif name == 'resnet50-ms-snn':
            backbone = ms_resnet50_snn(return_interm_indices=return_interm_indices, steps=steps)
            num_channels_all = [256, 512, 1024, 2048]
        elif name in ['spikenext-T', 'spikenext-L', 'spikenext-B']:
            backbone, num_channels_all = spikenext(modelname=name, out_indices=tuple(return_interm_indices),
                                                   steps=steps)
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))

        assert return_interm_indices in [[2, 3], [1, 2, 3], [0, 1, 2, 3]]

        num_channels = num_channels_all[4 - len(return_interm_indices):]
        super().__init__(backbone=backbone, train_backbone=train_backbone, num_channels=num_channels,
                         name=name)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone: 
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords: 
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    if args.backbone in ['resnet50-snn', 'resnet50-ms-snn', 'resnet50-ems-snn', 'resnet50-sew-snn', 'spikenext-T',
                         'spikenext-B', 'spikenext-L']:
        backbone = Backbone(name=args.backbone, train_backbone=train_backbone,
                            return_interm_indices=return_interm_indices, steps=args.STEPS_VALID)
        bb_num_channels = backbone.num_channels
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(
        type(bb_num_channels))
    return model


