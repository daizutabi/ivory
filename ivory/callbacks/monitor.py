from dataclasses import dataclass

import numpy as np

from ivory.core.state import State


@dataclass
class Monitor(State):
    metric: str = "val_loss"
    mode: str = "min"

    def __post_init__(self):
        self.is_best = False
        self.best_epoch = -1
        if self.mode == "min":
            self.best_score = np.inf
        elif self.mode == "max":
            self.best_score = -np.inf
        else:
            raise ValueError(f"mode must be 'min' or 'max': {self.mode} given.")

    def on_epoch_end(self, run):
        if self.metric not in run.metrics:
            msg = f"Metric '{self.metric}' not found. Available metrics are: "
            msg += f"{run.metrics}"
            raise ValueError(msg)
        score = run.metrics[self.metric]
        if self.mode == "min":
            self.is_best = score < self.best_score
        else:
            self.is_best = score > self.best_score
        if self.is_best:
            self.best_score = score
            self.best_epoch = run.metrics.epoch
        self.score = score
