from dataclasses import dataclass

import numpy as np

from ivory.core.dict import Dict
from ivory.core.state import State


@dataclass
class Metrics(Dict, State):
    def __post_init__(self):
        super().__post_init__()
        self.history = {}

    def __str__(self):
        metrics = []
        for metric in self:
            metrics.append(f"{metric}={self[metric]:.4g}")
        return " ".join(metrics)

    def __repr__(self):
        class_name = self.__class__.__name__
        args = str(self).replace(" ", ", ")
        return f"{class_name}({args})"

    def on_epoch_start(self, run):
        self.epoch = run.trainer.epoch

    def on_train_start(self, run):
        self.losses = []

    def on_val_start(self, run):
        self.losses = []

    def step(self, output, target):
        """Adds float loss to ``self.loss`` and returns a loss object (ex. tensor)."""
        pass

    def on_train_end(self, run):
        if self.losses:
            self["loss"] = np.mean(self.losses)

    def on_val_end(self, run):
        if self.losses:
            self["val_loss"] = np.mean(self.losses)

    def on_epoch_end(self, run):
        self.update(self.metrics_dict(run))
        self.update_history()

    def update_history(self):
        for metric, value in self.items():
            if metric not in self.history:
                self.history[metric] = {self.epoch: value}
            else:
                self.history[metric][self.epoch] = value

    def metrics_dict(self, run):
        """Returns an extra custom metrics dictionay."""
        return {}
