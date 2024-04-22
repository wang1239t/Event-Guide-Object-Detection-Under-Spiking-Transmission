from pathlib import Path
from datasets.coco import CocoDetection
import datasets.SOD.sod_transforms as T


def make_sod_transforms(setting, timestep, ts_v, use_pre_event):
    scales = [256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352]
    max_size = 633
    scales2_resize = [280, 300, 350]
    scales2_crop = [256, 350]
    mean = None
    std = None
    ts = timestep
    if ts_v == 2:
        mean = [0.20933848, 0.20905821, 0.30720114, 0.21023913, 0.20897032, 0.30761407]
        std = [0.1662545, 0.16642975, 0.24105527, 0.16807087, 0.16809275, 0.24339575]
    if ts_v == 3:
        mean = [0.20933848, 0.20905821, 0.30720114, 0.21023913, 0.20897032, 0.30761407, 0.20933848, 0.20905821,
                0.30720114, ]
        std = [0.1662545, 0.16642975, 0.24105527, 0.16807087, 0.16809275, 0.24339575, 0.1662545, 0.16642975, 0.24105527]
    if ts_v == 4:
        mean = [0.20933848, 0.20905821, 0.30720114, 0.21023913, 0.20897032, 0.30761407, 0.20933848, 0.20905821,
                0.30720114, 0.21023913, 0.20897032, 0.30761407]
        std = [0.1662545, 0.16642975, 0.24105527, 0.16807087, 0.16809275, 0.24339575, 0.1662545, 0.16642975, 0.24105527,
               0.16807087, 0.16809275, 0.24339575]

    if setting in ['train', 'train_low_light', 'train_motion_blur', 'train_normal']:
        return T.Compose([
            T.Drop(),
            T.Fuse(time_steps=ts, delta=110, steps_v=ts_v, pre_event=use_pre_event),
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([T.RandomResize(scales2_resize),
                           T.RandomSizeCrop(*scales2_crop),
                           T.RandomResize(scales, max_size=max_size),
                           ])),
            # T.Normalize([0.4589], [0.1005])
            T.Normalize(mean, std)
        ])
    elif setting in ['val', 'test', 'val_low_light', 'val_motion_blur', 'val_normal','test_low_light','test_motion_blur','test_normal']:
        return T.Compose([
            T.Fuse(time_steps=ts, delta=110, steps_v=ts_v, pre_event='VAL'),
            T.RandomResize([max(scales)], max_size=max_size),
            # T.Normalize([0.4589], [0.1005])
            T.Normalize(mean, std)
        ])
    raise ValueError(f'unknown {setting}')


def build(setting, args):
    root = Path(args.sod_path)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root / 'train' / 'events', root / 'train' / 'images',
                  root / 'train' / 'annotations' / 'train_annotations.json'),
        "val": (
            root / 'val' / 'events', root / 'val' / 'images', root / 'val' / 'annotations' / 'val_annotations.json'),
        "test": (
            root / 'test' / 'events', root / 'test' / 'images',
            root / 'test' / 'annotations' / 'test_annotations.json'),
        "test_low_light": (
            root / 'test_low_light' / 'events', root / 'test_low_light' / 'images',
            root / 'test_low_light' / 'annotations' / 'test_annotations.json'),
        "test_motion_blur": (
            root / 'test_motion_blur' / 'events', root / 'test_motion_blur' / 'images',
            root / 'test_motion_blur' / 'annotations' / 'test_annotations.json'),
        "test_normal": (
            root / 'test_normal' / 'events', root / 'test_normal' / 'images',
            root / 'test_normal' / 'annotations' / 'test_annotations.json'),
        "train_low_light": (root / 'train_low_light' / 'events', root / 'train_low_light' / 'images',
                            root / 'train_low_light' / 'annotations' / 'train_annotations.json'),
        "train_motion_blur": (root / 'train_motion_blur' / 'events', root / 'train_motion_blur' / 'images',
                              root / 'train_motion_blur' / 'annotations' / 'train_annotations.json'),
        "train_normal": (root / 'train_normal' / 'events', root / 'train_normal' / 'images',
                         root / 'train_normal' / 'annotations' / 'train_annotations.json'),
        "val_low_light": (root / 'val_low_light' / 'events', root / 'val_low_light' / 'images',
                          root / 'val_low_light' / 'annotations' / 'val_annotations.json'),
        "val_motion_blur": (root / 'val_motion_blur' / 'events', root / 'val_motion_blur' / 'images',
                            root / 'val_motion_blur' / 'annotations' / 'val_annotations.json'),
        "val_normal": (root / 'val_normal' / 'events', root / 'val_normal' / 'images',
                       root / 'val_normal' / 'annotations' / 'val_annotations.json'),
    }
    event_folder, image_folder, ann_file = PATHS[setting]

    dataset = CocoDetection(event_folder, image_folder, ann_file,
                            transforms=make_sod_transforms(setting, args.STEPS, args.STEPS_VALID,
                                                           args.use_pre_event),
                            return_masks=args.masks,
                            dataset_file=args.dataset_file)
    return dataset
