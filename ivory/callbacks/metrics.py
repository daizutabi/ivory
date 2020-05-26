"""Metrics to record scores while training."""
from dataclasses import dataclass
from typing import Any, Dict

import ivory.core.collections
from ivory.core.run import Run
from ivory.core.state import State


@dataclass
class Metrics(ivory.core.collections.Dict, State):
    """Metrics object."""

    def __post_init__(self):
        super().__post_init__()
        self.history = ivory.core.collections.Dict()

    def __str__(self):
        metrics = []
        for metric in self:
            metrics.append(f"{metric}={self[metric]:.4g}")
        return " ".join(metrics)

    def __repr__(self):
        class_name = self.__class__.__name__
        args = str(self).replace(" ", ", ")
        return f"{class_name}({args})"

    def on_epoch_begin(self, run: Run):
        if run.trainer:
            self.epoch = run.trainer.epoch
        else:
            self.epoch = 0

    def on_epoch_end(self, run: Run):
        self.update(self.metrics_dict(run))
        self.update_history()

    def update_history(self):
        for metric, value in self.items():
            if metric not in self.history:
                self.history[metric] = {self.epoch: value}
            else:
                self.history[metric][self.epoch] = value

    def metrics_dict(self, run: Run) -> Dict[str, Any]:
        """Returns an extra custom metrics dictionary."""
        return {}
