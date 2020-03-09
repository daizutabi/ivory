from typing import Any, Dict, Tuple

from torch import Tensor

import ivory.callbacks
from ivory.torch.utils import cpu


class Metrics(ivory.callbacks.Metrics):
    def evaluate(self, loss, output, target) -> Dict[str, float]:
        return {"loss": loss.item()}

    def train_evaluate(self, output, target) -> Tuple[Tensor, Dict[str, float]]:
        loss = self.criterion(output, target)
        output = output.detach()
        return loss, self.evaluate(loss, output, target)

    def val_evaluate(self, output, target) -> Tuple[Any, Dict[str, float]]:
        loss = self.criterion(output, target)
        output = output.detach()
        record = self.evaluate(loss, output, target)
        if output.device.type != "cpu":
            output = cpu(output)
        return output, record
