import sklearn.metrics

import ivory.callbacks.metrics


class Metrics(ivory.callbacks.metrics.Metrics):
    def call(self, output, target):
        pred = output.reshape(-1)
        true = target.reshape(-1)
        estimator = self.run.estimator.estimator
        metrics = {}
        if hasattr(estimator, "criterion"):
            if estimator.criterion == "mse":
                metrics["mse"] = sklearn.metrics.mean_squared_error(true, pred)
            elif estimator.criterion == "mae":
                metrics["mae"] = sklearn.metrics.mean_absolute_error(true, pred)
        return metrics
