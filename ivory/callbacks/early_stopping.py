from dataclasses import dataclass

from ivory.core.exceptions import EarlyStopped
from ivory.core.state import State


@dataclass
class EarlyStopping(State):
    """Early stop training loop when a metric has stopped improving.

    Args:
        patience (int): number of epochs with no improvement
            after which training will be stopped. Default: `0`.
        min_delta (float): minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: `0`.
    """

    patience: int = 0

    def __post_init__(self):
        self.wait = 0

    def on_epoch_end(self, run):
        if run.monitor.is_best:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                raise EarlyStopped
