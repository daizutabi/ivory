import numpy as np
import torch

import ivory.callbacks.results
from ivory.torch import utils


class Results(ivory.callbacks.results.Results):
    def reset(self):
        super().reset()
        self.indexes = []
        self.outputs = []
        self.targets = []

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

    def result_dict(self):
        index = np.concatenate(self.indexes)
        output = np.concatenate(self.outputs)
        if self.targets:
            target = np.concatenate(self.targets)
        else:
            target = None
        super().step(index, output, target)
        return super().result_dict()
