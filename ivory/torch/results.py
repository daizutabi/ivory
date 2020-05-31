import torch

import ivory.callbacks.results
from ivory.torch import utils


class Results(ivory.callbacks.results.BatchResults):
    def step(self, index, output, target=None):
        if torch.is_tensor(index):
            index = index.numpy()
        self.indexes.append(index)

        output = output.detach()
        if output.device.type != "cpu":
            output = utils.cpu(output)
        self.outputs.append(output.numpy())

        if target is not None:
            if torch.is_tensor(target):
                if target.device.type != "cpu":
                    target = utils.cpu(target)
                target = target.numpy()
            self.targets.append(target)
