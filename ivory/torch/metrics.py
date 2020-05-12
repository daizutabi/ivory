from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

import ivory.callbacks.metrics
from ivory.core.run import Run


@dataclass(repr=False)
class Metrics(ivory.callbacks.metrics.Metrics):
    criterion: Optional[Callable] = None

    def on_epoch_begin(self, run: Run):
        self.epoch = run.trainer.epoch

    def on_train_begin(self, run: Run):
        self.losses: List[float] = []

    def step(self, input, output, target):
        loss = self.criterion(output, target)
        self.losses.append(loss.item())
        return loss

    def on_train_end(self, run: Run):
        self["loss"] = np.mean(self.losses)

    def on_val_begin(self, run: Run):
        self.losses = []

    def on_val_end(self, run: Run):
        self["val_loss"] = np.mean(self.losses)

    def metrics_dict(self, run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}

    def save(self, state_dict, path):
        torch.save(state_dict, path)

    def load(self, path):
        return torch.load(path)
