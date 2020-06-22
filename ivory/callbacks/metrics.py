"""Metrics to record scores while training."""
from typing import List

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
        self.run = None

    def __str__(self):
        metrics = []
        for metric in self:
            metrics.append(f"{metric}={self[metric]:.4g}")
        return " ".join(metrics)

    def __repr__(self):
        class_name = self.__class__.__name__
        args = str(self).replace(" ", ", ")
        return f"{class_name}({args})"

    def __call__(self, output, target):
        metrics = {}
        for base in reversed(self.__class__.mro()[0:-3]):
            if hasattr(base, "call") and callable(base.call):
                metrics.update(base.call(self, output, target))
        for key, func in self.metrics_fn.items():
            metrics[key] = func(target, output)
        return metrics

    def on_init_begin(self, run: Run):
        self.run = run

    def on_epoch_begin(self, run: Run):
        if run.trainer:
            self.epoch = run.trainer.epoch
        else:
            self.epoch = 0

    def on_epoch_end(self, run: Run):
        val = run.results.val
        metrics = self(val.output, val.target)
        self.update(metrics)
        self.update_history()

    def update_history(self):
        for metric, value in self.items():
            if metric not in self.history:
                self.history[metric] = {self.epoch: value}
            else:
                self.history[metric][self.epoch] = value


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

    def on_epoch_end(self, run: Run):
        val = run.results.val
        metrics = self(val.output, val.target)
        del metrics["loss"]
        self.update(metrics)
        self.update_history()

    def call(self, output, target):
        return {"loss": np.mean(self.losses)}


METRICS = {"mse": "sklearn.metrics.mean_squared_error"}


def get_metric_function(key, value):
    if value is None:
        if key not in METRICS:
            raise ValueError(f"Unkown metric: {key}")
        value = METRICS[key]
    if isinstance(value, str) and "." not in value:
        value = f"sklearn.metrics.{value}"

    return instance.get_attr(value)
