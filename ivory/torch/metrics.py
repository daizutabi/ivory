from typing import Callable

import ivory.callbacks.metrics
from ivory.torch import utils


class Metrics(ivory.callbacks.metrics.Metrics):
    def __init__(self, criterion: Callable):
        super().__init__()
        self.criterion = criterion

    def train_evaluate(self, index, output, target):
        loss = self.criterion(output, target)
        return loss, loss.item()

    def val_evaluate(self, index, output, target):
        loss = self.criterion(output, target)
        output = output.detach()
        if output.device.type != "cpu":
            output = utils.cpu(output)
            target = utils.cpu(target)
        return index.numpy(), output.numpy(), target.numpy(), loss.item()

    def test_evaluate(self, index, output):
        output = output.detach()
        if output.device.type != "cpu":
            output = utils.cpu(output)
        return index.numpy(), output.numpy()

    def record_dict(self, run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}
