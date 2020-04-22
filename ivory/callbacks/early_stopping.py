from dataclasses import dataclass

from ivory.core.exceptions import EarlyStopped
from ivory.core.state import State


@dataclass
class EarlyStopping(State):
    """Early stops a training loop when a metric has stopped improving.

    Args:
        patience (int): number of epochs with no improvement
            after which training will be stopped. Default: `0`.
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
