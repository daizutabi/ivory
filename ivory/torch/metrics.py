from dataclasses import dataclass
from typing import Callable, Optional

import ivory.callbacks.metrics


@dataclass(repr=False)
class Metrics(ivory.callbacks.metrics.Metrics):
    criterion: Optional[Callable] = None

    def step(self, output, target):
        loss = self.criterion(output, target)
        self.losses.append(loss.item())
        return loss

    def metrics_dict(self, run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}
