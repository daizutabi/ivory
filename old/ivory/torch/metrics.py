import ivory.callbacks.metrics


class Metrics(ivory.callbacks.metrics.BatchMetrics):
    def call(self, output, target):
        return {"lr": self.run.optimizer.param_groups[0]["lr"]}

    # def save(self, state_dict, path):
    #     torch.save(state_dict, path)
    #
    # def load(self, path):
    #     return torch.load(path)
