from dataclasses import dataclass

import numpy as np

from ivory.core.run import Run
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
            raise ValueError(f"Mode must be 'min' or 'max': {self.mode!r} given.")

    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"metric={self.metric!r}, mode={self.mode!r}"
        if self.best_epoch != -1:
            s += f", best_score={self.best_score:.3g}, best_epoch={self.best_epoch}"
        return f"{class_name}({s})"

    def on_epoch_end(self, run: Run):
        if self.metric not in run.metrics:
            msg = f"Metric {self.metric!r} not found. Available metrics: {run.metrics}"
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
