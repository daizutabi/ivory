import numpy as np
from dataclasses import dataclass

from ivory.callbacks import Callback


@dataclass
class EarlyStopping(Callback):
    """Early stop training loop when a metric has stopped imporving.

    Args:
        monitor (str): quantity to be monitored. Default: `'val_loss'`.
        mode (str): one of `min`, `max`. In `min` mode, training will
            stop when the quantity monitored has stopped decreasing;
            in `max` mode it will stop when the quantity monitored has
            stopped increasing. Default: `min`.
        patience (int): number of epochs with no improvement
            after which training will be stopped. Default: `0`.
        min_delta (float): minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: `0`.
    """

    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 0
    min_delta: float = 0.0

    @property
    def is_best(self):
        if self.mode == "min":
            return self.current_score + self.min_delta < self.best_score
        else:
            return self.current_score - self.min_delta > self.best_score

    def on_fit_start(self, run):
        self.wait = 0
        self.best_score = np.inf if self.mode == "min" else -np.inf

    def on_epoch_end(self, run):
        self.current_score = run.metrics.current_record[self.monitor]
        if self.is_best:
            self.best_score = self.current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                raise StopIteration
