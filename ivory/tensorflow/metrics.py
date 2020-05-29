from tensorflow.python.keras import backend as K

import ivory.callbacks.metrics
from ivory.core.run import Run


class Metrics(ivory.callbacks.metrics.Metrics):
    def metrics_dict(self, run: Run):
        return {"lr": float(K.get_value(run.model.optimizer.lr))}
