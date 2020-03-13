import ivory.callbacks
from ivory.torch import utils


class Metrics(ivory.callbacks.Metrics):
    def criterion(self, output, target):
        """Returns loss tensor."""
        raise NotImplementedError

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

    def record_dict(self, run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}
