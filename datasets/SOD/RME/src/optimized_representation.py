import os
import random
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "representations"))
sys.path.append(os.path.join(ROOT, "ev-YOLOv6/"))

from datasets.SOD.RME.src.mixed_density_event_stack import (
    MixedDensityEventStack,
)

N_CHANNELS = 5

class RMEToTensor(object):
    def __init__(self, time_steps, pre_event='RME'):
        self.height = 260
        self.width = 346
        self.time_steps = time_steps
        self.pre_event = pre_event

    def __call__(self, event):
        if isinstance(event, str): event = np.load(event)
        num_events = len(event)
        event1 = torch.from_numpy(
            get_optimized_representation(event, num_events, self.height, self.width, self.time_steps, self.pre_event)[
                0]).permute(2, 0,
                            1)
        event2 = torch.from_numpy(
            get_optimized_representation(event, num_events, self.height, self.width, self.time_steps, self.pre_event)[
                0]).permute(2, 0,
                            1)
        return event1, event2


def get_optimized_representation(reshaped_return_data, num_events, height, width, time_steps, pre_event):
    window_indexes = list(range(time_steps))

    measure_functions = [
        "polarity",
        "timestamp_neg",
        "count_neg",
        "count_pos",
        "count",
        "timestamp_pos",
        "timestamp",
    ]
    aggregations_functions = [
        "variance",
        "sum",
        "mean",
        "max",
    ]

    stack_size = time_steps
    functions = list(random.choices(measure_functions, k=time_steps))
    aggregations = list(random.choices(aggregations_functions, k=time_steps))

    if pre_event == 'TEST':
        functions = [
            "timestamp",
            "timestamp_pos",
            "timestamp_neg",
            "timestamp_pos",
            "timestamp_neg"]
        aggregations = [
            "sum",
            "sum",
            "sum",
            "sum",
            "sum"]

    if pre_event == 'VAL':
        functions = [
            "timestamp",
            "timestamp_pos",
            "timestamp_neg",
            "count_neg",
            "count_pos"]
        aggregations = [
            "max",
            "sum",
            "mean",
            "sum",
            "mean"]

    stacking_type = "SBT"

    indexes_functions_aggregations = window_indexes, functions, aggregations

    transformation = MixedDensityEventStack(
        stack_size,
        num_events,
        height,
        width,
        indexes_functions_aggregations,
        stacking_type,
    )
    rep = transformation.stack(reshaped_return_data)

    return rep


if __name__ == "__main__":
    path = "E:/Gen1_NPY/Gen1_npy/train/17-10-12_15-54-21_305500000_365500000_8749999.npy"
    save_path = 'D:/data/GEN1-Deal'
    reshaped_return_data = np.load(path)
    rep, operation_names = get_optimized_representation(reshaped_return_data, num_events=30000, height=240, width=304)
    print(operation_names)
    for channel in range(rep.shape[2]):
        channel_data = rep[:, :, channel]

        plt.imshow(channel_data, cmap='gray')
        plt.title(f'Channel {channel + 1}')
        plt.show()
