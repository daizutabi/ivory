import ivory.callbacks
from ivory.torch import utils


class Metrics(ivory.callbacks.Metrics):
    def evaluate(self, loss, output, target):
        return {"loss": loss.item()}

    def train_evaluate(self, index, output, target):
        loss = self.criterion(output, target)
        output = output.detach()
        return loss, self.evaluate(loss, output, target)

    def val_evaluate(self, index, output, target):
        loss = self.criterion(output, target)
        output = output.detach()
        record = self.evaluate(loss, output, target)
        if output.device.type != "cpu":
            output = utils.cpu(output)
        return index.numpy(), output.numpy(), record

    def on_current_record(self, run):
        self.current_record["lr"] = run.optimizer.param_groups[0]["lr"]
