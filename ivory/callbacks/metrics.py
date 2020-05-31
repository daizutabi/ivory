"""Metrics to record scores while training."""
from typing import Any, Dict, List

import numpy as np

import ivory.core.collections
from ivory.core import instance
from ivory.core.run import Run
from ivory.core.state import State


class Metrics(ivory.core.collections.Dict, State):
    """Metrics object."""

    def __init__(self, **kwargs):
        super().__init__()
        self.metrics_fn = {}
        for key, value in kwargs.items():
            self.metrics_fn[key] = get_metric_function(key, value)
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
        pred = run.results.val.output.reshape(-1)
        true = run.results.val.target.reshape(-1)
        metrics_dict = {}
        for key, func in self.metrics_fn.items():
            metrics_dict[key] = func(true, pred)
        return metrics_dict


class BatchMetrics(Metrics):
    def on_epoch_begin(self, run: Run):
        self.epoch = run.trainer.epoch

    def on_train_begin(self, run: Run):
        self.losses: List[float] = []

    def step(self, loss: float):
        self.losses.append(loss)

    def on_train_end(self, run: Run):
        self["loss"] = np.mean(self.losses)

    def on_val_begin(self, run: Run):
        self.losses = []

    def on_val_end(self, run: Run):
        self["val_loss"] = np.mean(self.losses)


METRICS = {"mse": "sklearn.metrics.mean_squared_error"}


def get_metric_function(key, value):
    if value is None:
        if key not in METRICS:
            raise ValueError(f"Unkown metric: {key}")
        value = METRICS[key]
    if isinstance(value, str) and "." not in value:
        value = f"sklearn.metrics.{value}"

    return instance.get_attr(value)
