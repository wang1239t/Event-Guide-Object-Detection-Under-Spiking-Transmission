# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries，正负样本
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]  # 查明几个标签，存成一个list，每个张量是每张图的标签数
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))  # 根据现有的标签数量生成噪声框
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])  # 找到所有标签类别，batch为一个整体
        boxes = torch.cat([t['boxes'] for t in targets])  # 找到所有框坐标
        batch_idx = torch.cat(
            [torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])  # 对每个标签判断是哪个batch的
        # 找到unmask_label + unmask_bbox中非零元素的索引，并将这些索引展平为一维张量known_indice
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(
            -1)  # 原来的known_indice重复2 * dn_number次，并按顺序排列，形成320维tensor
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)  # 对label进行上述复制
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)  # 由于boxes为（10，4），复制完成为（320，4）
        known_labels_expaned = known_labels.clone()  # 存储扩展后的known_labels_expaned和known_bbox_expand
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())  # 生成一个随机分布的张量
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1)  # half of bbox prob # 选出小于p < (label_noise_ratio * 0.5)的索引
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here # 随机生成标签
            known_labels_expaned.scatter_(0, chosen_indice,
                                          new_label)  # 使用新生成的标签 new_label 将 known_labels_expaned 张量中对应索引
            # chosen_indice 的元素进行替换。这里的 scatter_ 函数用于原地替换操作，将新标签放置到 chosen_indice 对应的位置上。
        single_pad = int(max(known_num))  # 用于后续的填充操作

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number,
                                                                                         1)  # 生成一个形状为 (dn_number,
        # len(boxes)) 的张量
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()  # 生成正负标签索引（每10个box交替替换，正为1-10，21-30,...,负为11-20，31-40,...）
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2  # 计算known_bbox的左上角和右上角坐标，并存储到known_bbox_中

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2  # 把known_bboxs的w，h除以2，分两次存在diff中
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2,
                                           dtype=torch.float32) * 2.0 - 1.0  # 随机生成要么是1，要么是-1的随机数
            rand_part = torch.rand_like(
                known_bboxs)  # 创建一个与 known_bboxs 张量形状相同的随机数张量 rand_part。这个随机数张量的值在区间 [0, 1) 内均匀分布
            rand_part[negative_idx] += 1.0  # 这个操作将 rand_part 张量中与负样本索引对应的位置的元素增加 1.0，负样本让其远离原始框
            rand_part *= rand_sign  # 对 rand_part 张量中的元素进行正负翻转。如果 rand_sign 张量的元素为 -1.0
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale  # 这个操作会根据随机数和缩放因子来对已知边界框进行扰动，将已知边界框
            # known_bbox_ 与生成的随机数 rand_part 和 diff 进行逐元素相乘，并乘以 box_noise_scale，然后加到 known_bbox_ 上
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)  # 对更新后的边界框 known_bbox_ 进行裁剪，将其限制在区间 [0.0, 1.0] 内
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]  # 更新known_bbox_expand

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord
