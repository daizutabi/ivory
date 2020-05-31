import ivory.callbacks.metrics
from ivory.core.run import Run


class Metrics(ivory.callbacks.metrics.BatchMetrics):
    def metrics_dict(self, run: Run):
        return {}
