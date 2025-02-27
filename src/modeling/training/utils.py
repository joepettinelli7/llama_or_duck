from __future__ import print_function
import datetime
import logging
import time
from collections import defaultdict
from collections import deque
from typing import Generator
from typing import Optional
import torch
import torch.distributed as dist


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed
    values over a window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        """Create the queue and initialize values."""
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        """
        Append value to the queue.

        Args:
            value: The value to update with (mostly BCE)
            n:

        Returns:

        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        """
        Return the median of the elements in the queue.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        """
        Return the mean of the elements in the queue.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        """
        Return the aggregate mean from the total value.
        """
        return self.total / self.count

    @property
    def max(self) -> float:
        """
        Return the max value in the queue.
        """
        return max(self.deque)

    @property
    def value(self) -> float:
        """
        Return the last element of the queue.
        """
        return self.deque[-1]

    def __str__(self) -> str:
        """
        Format the value to provide mean, average, global average,
        # max and last value.
        """
        return self.fmt.format(
            median=self.median, avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    """
    Log metrics.
    """

    def __init__(self, delimiter: str = "\t", mode: str = "train"):
        """
        Save the parameters.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.mode = mode

    def update(self, **kwargs):
        """
        Update the underlying dict of metrics.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        Return the attribute first from the underlying dict of metrics,
        then the object dict.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        """
        Concatenate all serialized metric values.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(self.mode + "_" + name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name: str, meter: SmoothedValue):
        """
        Add a metric to the underlying dict.
        """
        self.meters[name] = meter

    def log_every(
        self,
        iterable: torch.utils.data.DataLoader,  # ignore
        header: Optional[str] = None,
        logger: logging.Logger = None
    ) -> Generator[tuple, None, None]:
        """
        Yield batched data and log data after print_freq iterations.
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ]
        )
        mb = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                msg = log_msg.format(
                    i,
                    len(iterable),
                    eta=eta_string,
                    meters=str(self),
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / mb,
                )
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = "{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable))
        if logger:
            logger.info(msg)
        else:
            print(msg)


def warmup_lr_scheduler(optimizer, warmup_iters: int, warmup_factor: float):
    """
    Set the learning rate for each parameter group.
    """
    print("warmup", type(optimizer), type(warmup_iters))

    def f(x: int):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def is_dist_avail_and_initialized():
    """
    Check if distributed training is feasible.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Returns the number of processes in the current process group.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
