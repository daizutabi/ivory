from dataclasses import dataclass
from typing import Callable

import ivory.callbacks.metrics


@dataclass(repr=False)
class Metrics(ivory.callbacks.metrics.Metrics):
    criterion: Callable

    def step(self, output, target):
        loss = self.criterion(output, target)
        self.losses.append(loss.item())
        return loss

    def metrics_dict(self, run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}
