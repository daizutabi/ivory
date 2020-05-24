import numpy as np

import ivory.callbacks.results
from ivory.torch import utils


class Results(ivory.callbacks.results.Results):
    def reset(self):
        super().reset()
        self.indexes = []
        self.outputs = []
        self.targets = []

    def step(self, index, output, target=None):
        self.indexes.append(index.numpy())

        output = output.detach()
        if output.device.type != "cpu":
            output = utils.cpu(output)
        self.outputs.append(output.numpy())

        if target is not None:
            if target.device.type != "cpu":
                target = utils.cpu(target)
            self.targets.append(target.numpy())

    def result_dict(self):
        index = np.concatenate(self.indexes)
        output = np.concatenate(self.outputs)
        if self.targets:
            target = np.concatenate(self.targets)
        else:
            target = None
        super().step(index, output, target)
        return super().result_dict()
