import ivory.callbacks.metrics
from ivory.core.run import Run


class Metrics(ivory.callbacks.metrics.BatchMetrics):
    def metrics_dict(self, run: Run):
        return {"lr": run.optimizer.param_groups[0]["lr"]}

    # def save(self, state_dict, path):
    #     torch.save(state_dict, path)
    #
    # def load(self, path):
    #     return torch.load(path)
