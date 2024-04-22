import argparse
import os
import random
import sys
import warnings

import cv2
import json
import numpy as np
import torch

from datasets.SOD.sod import make_sod_transforms
from models.dino.dino import build_dino

import util.misc as utils
from config import get_args_parser
from util import box_ops
from util.logger import setup_logger
from util.slconfig import SLConfig

warnings.filterwarnings("ignore", category=UserWarning)

CLASSES = ['car', 'pedestrian', 'two-wheeler']


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 2
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect(args, directory, save_directory):
    utils.init_distributed_mode(args)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:  # 存入所有设置
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False,
                          name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + ' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    model, criterion, postprocessors = build_dino(args)
    model.to(device)
    ema_m = None

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    logger.info(
        "params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m

    model_without_ddp.to(device)
    model_without_ddp.eval()
    criterion.eval()
    target = None

    for filename in os.listdir(directory):
        data = os.path.splitext(filename)[0]
        data = data[:-2]
        png = f'/home/dataset/SOD/train/images/{data}.png'
        npy = f'/home/dataset/SOD/train/events/{data}.npy'
        detect_png_path = os.path.join(directory, filename)
        detect_save_png_path = os.path.join(save_directory, '{}.png'.format(data))

        transforms = make_sod_transforms('test', args.STEPS, args.STEPS_VALID,
                                         args.use_pre_event)
        img, evt, target = transforms(png, npy, target)
        NH = img.shape[-2]
        NW = img.shape[-1]
        img_list = []
        img_list.append(img)
        n_img = utils.nested_tensor_from_tensor_list(img_list)
        output = model_without_ddp(n_img.to(device))
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

        image = cv2.imread(detect_png_path)

        thershold = 0.40
        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > thershold

        tgt = {
            'boxes': boxes[select_mask],
            'size': torch.Tensor([NH, NW]),
            'orig_size': [260, 346],
            'box_label': labels[select_mask],
            'conf': scores[select_mask]
        }

        H, W = tgt['size'].tolist()
        Ho, Wo = tgt['orig_size']
        for i in range(len(tgt['box_label'])):
            unnormbbox = tgt['boxes'][i].cpu() * torch.Tensor([Wo, Ho, Wo, Ho])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            unnormbbox[2:] += unnormbbox[:2]
            xyxy = unnormbbox.numpy()

            label_num = tgt['box_label'][i]
            conf = tgt['conf'][i]
            if int(label_num) == 0:
                label, color = 'car', (255, 0, 0)
            elif int(label_num) == 1:
                label, color = 'pedestrian', (0, 0, 255)
            elif int(label_num) == 2:
                label, color = 'two-wheeler', (0, 255, 0)
            lab_and_conf = f'{label:} {conf:.2f}'
            plot_one_box(xyxy, image, label=lab_and_conf, color=color, line_thickness=1)

        # cv2.imshow('detect', image)
        cv2.imwrite(detect_save_png_path, image)
        print('successful')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    options = {'dn_scalar': 100, 'embed_init_tgt': True, 'dn_label_coef': 1.0, 'dn_bbox_coef': 1.0, 'use_ema': False,
               'dn_box_noise_scale': 1.0}
    args.options = options
    # directory代表需要检测的图片位置
    # save_directory代表检测结束保存的图片位置
    directory = '/home/wanghechong/dataset/train_normal/fuse_img_normal'
    save_directory = '/home/wanghechong/dataset/train_normal/fuse_img_det_normal'
    detect(args, directory, save_directory)
