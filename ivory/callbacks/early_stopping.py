from dataclasses import dataclass


@dataclass
class EarlyStopping:
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

    patience: int = 0

    def __post_init__(self):
        self.wait = 0

    def on_epoch_end(self, run):
        if run.monitor.is_best:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                raise StopIteration
