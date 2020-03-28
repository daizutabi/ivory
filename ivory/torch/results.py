import ivory.callbacks.results
from ivory.torch import utils


class Results(ivory.callbacks.results.Results):
    def step(self, index, output, target=None):
        output = output.detach()
        if output.device.type != "cpu":
            output = utils.cpu(output)
            if target is not None:
                target = utils.cpu(target)
        self.indexes.append(index.numpy())
        self.outputs.append(output.numpy())
        if target is not None:
            self.targets.append(target.numpy())
