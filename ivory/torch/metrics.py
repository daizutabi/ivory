from typing import List

import numpy as np

import ivory.callbacks.metrics
from ivory.core.run import Run


class Metrics(ivory.callbacks.metrics.Metrics):
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

    def metrics_dict(self, run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}

    # def save(self, state_dict, path):
    #     torch.save(state_dict, path)
    #
    # def load(self, path):
    #     return torch.load(path)
