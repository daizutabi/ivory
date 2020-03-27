import ivory.callback.results
from ivory import utils


class Results(ivory.callback.results.Results):
    def step(self, index, output, target=None):
        output = output.detach()
        if output.device.type != "cpu":
            output = utils.cpu(output)
            if target:
                target = utils.cpu(target)
        self.indexes.append(index.numpy())
        self.outputs.append(output.numpy())
        if target is not None:
            self.targets.append(target.numpy())
