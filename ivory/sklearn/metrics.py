import sklearn.metrics

import ivory.callbacks.metrics


class Metrics(ivory.callbacks.metrics.Metrics):
    def metrics_dict(self, run):
        pred = run.results.val.output.reshape(-1)
        true = run.results.val.target.reshape(-1)
        estimator = run.estimator.estimator
        metrics_dict = {}
        if hasattr(estimator, "criterion"):
            if estimator.criterion == "mse":
                metrics_dict["mse"] = sklearn.metrics.mean_squared_error(true, pred)
            elif estimator.criterion == "mae":
                metrics_dict["mae"] = sklearn.metrics.mean_absolute_error(true, pred)
        metrics_dict.update(super().metrics_dict(run))
        return metrics_dict
