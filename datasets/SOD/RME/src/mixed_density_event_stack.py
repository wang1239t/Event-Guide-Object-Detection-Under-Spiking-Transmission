# author: Nikola Zubic

import numpy as np
from datasets.SOD.RME.src.operations import Operations


class MixedDensityEventStack(object):
    def __init__(
            self,
            stack_size,
            num_of_events,
            height,
            width,
            indexes_functions_aggregations,
            stacking_type,
    ):
        self.stack_size = stack_size
        self.num_of_events = num_of_events
        self.height = height
        self.width = width
        self.indexes_functions_aggregations = indexes_functions_aggregations
        self.stacking_type = stacking_type

    def stack(self, event_sequence):
        x = event_sequence[:, 0].astype(np.int32)
        y = event_sequence[:, 1].astype(np.int32)
        p = event_sequence[:, 3].astype(np.int32)
        t = event_sequence[:, 2].astype(np.int64)

        assert len(x) == len(y) == len(p) == len(t)

        t = t - t.min()
        stacked_events = self.make_stack(x, y, p, t)

        rep = np.zeros((self.height, self.width, self.stack_size), dtype=np.float64)
        operation_names = []

        for i in range(len(stacked_events)):
            stacked_event = stacked_events[i]
            operation_name = next(iter(stacked_event))
            event_surface = stacked_event[operation_name]
            rep[:, :, i] = event_surface
            operation_names.append(operation_name)

        return rep, operation_names

    def create_windows(self, x, y, p, t):
        """
        Create 4 windows with halving strategy.
        Create 3 windows with equispaced strategy.
        This gives in total 7 different windows that are used for stacking, and effectively the strategy is mixed.
        """
        windows = []
        equispaced_factor = 1 / int(self.stack_size)
        windows.append((x, y, p, t))

        if self.stacking_type == "SBT":
            for i in range(int(self.stack_size)-1):
                x_ = x[
                    np.logical_and(
                        t <= (i + 1) * equispaced_factor, t >= i * equispaced_factor
                    )
                ]
                y_ = y[
                    np.logical_and(
                        t <= (i + 1) * equispaced_factor, t >= i * equispaced_factor
                    )
                ]
                p_ = p[
                    np.logical_and(
                        t <= (i + 1) * equispaced_factor, t >= i * equispaced_factor
                    )
                ]
                t_ = t[
                    np.logical_and(
                        t <= (i + 1) * equispaced_factor, t >= i * equispaced_factor
                    )
                ]
                windows.append((x_, y_, p_, t_))
            return windows
        raise ValueError(f'stacking_type is error ')

    def make_stack(self, x, y, p, t):
        t = t - t.min()
        time_interval = t.max() - t.min()
        t_s = t / time_interval
        stacked_event_list = []
        windows = self.create_windows(x, y, p, t_s)
        for i in range(self.stack_size):
            try:
                stacked_event = self.stack_data(
                    *windows[self.indexes_functions_aggregations[0][i]],
                    func=self.indexes_functions_aggregations[1][i],
                    aggregation=self.indexes_functions_aggregations[2][i]
                )
            except:
                stacked_event = {"": np.zeros((self.height, self.width))}
            stacked_event_list.append(stacked_event)

        return stacked_event_list

    def stack_data(self, x, y, p, t_s, func, aggregation):
        assert len(x) == len(y) == len(p) == len(t_s)

        events = np.concatenate(
            [
                x[..., np.newaxis],
                y[..., np.newaxis],
                t_s[..., np.newaxis],
                p[..., np.newaxis],
            ],
            axis=1,
        )

        event_surface = Operations(func, aggregation, self.height, self.width)(events)

        event_surface = {
            "_".join([func.capitalize(), aggregation.capitalize()]): event_surface
        }

        return event_surface
