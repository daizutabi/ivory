import ivory.callbacks.results
from ivory.torch import utils


class Results(ivory.callbacks.results.Results):
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
