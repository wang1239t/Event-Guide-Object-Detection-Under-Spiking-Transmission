import os.path
import random

import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
import numpy as np
from typing import Any

from datasets.SOD.RME.src.optimized_representation import RMEToTensor
from util.box_ops import box_xyxy_to_cxcywh
import torchvision.transforms.functional as F
from util.misc import interpolate
import torchvision.transforms as T
from datasets.SOD.EDI.edi import blur_to_sharp


@torch.jit.unused
def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


@torch.jit.unused
def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def crop(image, event, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    target["size"] = torch.tensor([h, w])
    fields = ["labels", "area", "iscrowd"]
    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)
        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, event, target


def resize(image, event, target, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        h, w = image_size
        if h == size:
            return (h, w)
        oh = size
        ow = int(size * w / h)
        return [oh, ow]

    def get_size(image_size, size, max_size=None):
        return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size((image.shape[1], image.shape[2]), size, max_size)
    rescaled_image = F.resize(image, size)
    rescaled_image_size = (rescaled_image.shape[1], rescaled_image.shape[2])

    if target is None:
        return rescaled_image, event, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image_size, (image.shape[1], image.shape[2])))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, event, target


def hflip(image, event, target):
    flipped_image = F.hflip(image)
    h, w = flipped_image.shape[1], flipped_image.shape[2]
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])  # xyxy的垂直镜像反转
        target["boxes"] = boxes
    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    return flipped_image, event, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, event, target):
        if random.random() < self.p:
            return hflip(image, event, target)
        return image, event, target


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, event, target):
        if random.random() < self.p:
            return self.transforms1(image, event, target)
        return self.transforms2(image, event, target)


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, event, target):
        size = random.choice(self.sizes)
        return resize(image, event, target, size, self.max_size)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, event, target):
        img_height, img_width = image.shape[1], image.shape[2]
        w = random.randint(self.min_size, min(img_width, self.max_size))
        h = random.randint(self.min_size, min(img_height, self.max_size))
        region = T.RandomCrop.get_params(image, (h, w))
        return crop(image, event, target, region)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, event, target=None):
        image /= 255
        mean = self.mean
        std = self.std
        image = F.normalize(image, mean=mean, std=std)

        if target is None:
            return image, event, None
        target = target.copy()
        h, w = image.shape[1], image.shape[2]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, event, target


class Drop(object):
    def __init__(self, d=0.1, p=0.5):
        self.d = d
        self.p = p

    def __call__(self, image, event, target=None):
        data = np.load(event)
        if random.random() < self.p:
            data_num = data.shape[0]
            del_num = int(data_num * self.d)
            indices = np.random.choice(data_num, size=del_num, replace=False)
            data = np.delete(data, indices, axis=0)
        event = data
        return image, event, target


class Fuse(object):
    def __init__(self, time_steps=5, delta=100, steps_v=4, pre_event='RME'):
        self.steps = time_steps
        self.s_v = steps_v
        self.delta = delta
        self.RME = RMEToTensor(time_steps=time_steps, pre_event=pre_event)
        self.pre_event = pre_event
        self.H = 260
        self.W = 346

    def __call__(self, image, event, target) -> (Tensor, Any, Any):
        default_float_dtype = torch.get_default_dtype()
        EDI_imgs, deltaT, delta = blur_to_sharp(image, event, v_length=self.steps, delta=self.delta)
        if self.pre_event == 'RME':
            pos_img, neg_img = self.RME(event)
        elif self.pre_event == 'VAL':
            pos_img, neg_img = self.RME(event)
        else:
            raise Exception("Unknown pre deal way")
        viewtensor(pos_img, False)
        viewtensor(neg_img, False)
        f_img = np.zeros((self.s_v, 3, self.H, self.W))
        for i in range(self.steps - self.s_v, self.steps):
            edi = np.squeeze(EDI_imgs[i])
            edi = edi / 255
            pos = pos_img[i].numpy()
            neg = neg_img[i].numpy()
            f_img[i - self.steps][0] = edi * 2 / 3 + neg * 1 / 3
            f_img[i - self.steps][1] = edi * 2 / 3 + pos * 1 / 3
            f_img[i - self.steps][2] = edi
        STEPS, C, H, W = f_img.shape
        f_img = f_img * 255
        image = torch.from_numpy(f_img).view((STEPS * C, H, W)).to(torch.float32)

        if isinstance(image, torch.ByteTensor):
            return image.to(dtype=default_float_dtype), event, target
        else:
            return image, event, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, event, target):
        for t in self.transforms:
            image, event, target = t(image, event, target)
        return image, event, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def viewtensor(out_dense, run=False):
    if run:
        # 遍历每个图像并显示
        for i in range(out_dense.shape[0]):
            # 将张量从 Torch 张量转换为 NumPy 数组
            # 获取非零位置的值
            values = nonzero_values(out_dense[i])

            # 打印非零位置的值
            print(values)
            array = out_dense[i].numpy()

            # 使用 plt.imshow() 函数显示灰度图
            plt.imshow(array, cmap='gray')

            # 隐藏坐标轴
            plt.axis('off')

            # 显示图像
            plt.show()
    pass


def nonzero_values(tensor):
    indices = torch.nonzero(tensor)
    values = tensor[indices[:, 0], indices[:, 1]]
    return values


class Event(object):
    def __init__(self, time_steps=5, pre_event='TEST'):
        self.steps = time_steps
        self.RME = RMEToTensor(time_steps=time_steps, pre_event=pre_event)
        self.pre_event = pre_event
        self.H = 260
        self.W = 346

    def __call__(self, event):
        pos_img, neg_img = self.RME(event)
        return pos_img


if __name__ == '__main__':
    directory = '/home/wanghechong/dataset/train_normal/events'
    for filename in os.listdir(directory):
        data = os.path.splitext(filename)[0]
        npy_path = f'/home/dataset/SOD/train/events/{data}.npy'
        event = Event(time_steps=5, pre_event='TEST')
        image = event(npy_path)
        T, H, W = image.shape
        image = image.permute(1, 2, 0)*255
        image1 = image[..., 3].numpy().astype(np.uint8)
        image2 = image[..., 4].numpy().astype(np.uint8)
        save_png_path1 = os.path.join('/home/wanghechong/dataset/train_normal/event_image_t_normal',
                                      '{}_1.png'.format(data))
        save_png_path2 = os.path.join('/home/wanghechong/dataset/train_normal/event_image_t_normal',
                                      '{}_2.png'.format(data))
        # 将ndarray转换为PIL图像
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        # 保存图像文件
        image1.save(save_png_path1)
        image2.save(save_png_path2)
        print('successful')
        # print(data)
        # C_, H, W = image.shape
        # x = image.view(-1, 3, H, W)
        # x_1 = x[0].numpy()
        # x_2 = x[-1].numpy()
        #
        # from PIL import Image
        # f_img_pil = Image.fromarray((x_2).astype(np.uint8).transpose(1, 2, 0))
        # f_img_pil.show()
