from dataclasses import dataclass

import numpy as np

from ivory.core.state import State


@dataclass
class Monitor(State):
    monitor: str = "val_loss"
    mode: str = "min"
    min_delta: float = 0.0

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
        self.score = run.metrics.record[self.monitor]
        if self.mode == "min":
            self.is_best = self.score < self.best_score - self.min_delta
        elif self.mode == "max":
            self.is_best = self.score > self.best_score + self.min_delta
        if self.is_best:
            self.best_score = self.score
            self.best_epoch = run.metrics.epoch
