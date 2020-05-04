from dataclasses import dataclass
from typing import Callable, Optional

import torch

import ivory.callbacks.metrics


@dataclass(repr=False)
class Metrics(ivory.callbacks.metrics.Metrics):
    criterion: Optional[Callable] = None

    def step(self, input, output, target):
        loss = self.criterion(output, target)
        self.losses.append(loss.item())
        return loss

    def metrics_dict(self, run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}

    def save(self, state_dict, path):
        torch.save(state_dict, path)

    def load(self, path):
        return torch.load(path)
