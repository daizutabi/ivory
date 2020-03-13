import os
import shutil
import tempfile
import time

import mlflow
import yaml
from mlflow.entities import Metric, Param
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from ivory.callbacks import Callback


class Tracking(Callback):
    @classmethod
    def on_experiment_start(cls, experiment):
        cls.client = mlflow.tracking.MlflowClient()
        exp = cls.client.get_experiment_by_name(experiment.name)
        if exp is not None:
            cls.experiment_id = exp.experiment_id
        else:
            cls.experiment_id = cls.client.create_experiment(experiment.name)
        experiment.experiment_id = cls.experiment_id

    def on_fit_start(self, run):
        self.directory = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.directory, "current"))
        tags = context_registry.resolve_tags({MLFLOW_RUN_NAME: run.name})
        r = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = r.info.run_id
        self.log_params(run.params)
        path = os.path.join(self.directory, "params.yaml")
        with open(path, "w") as file:
            yaml.dump(run.params, file, sort_keys=False)

    def on_epoch_end(self, run):
        self.log_metrics(dict(run.metrics.record), run.metrics.epoch)
        src = os.path.join(self.directory, "current")
        run.save(src)
        if run.monitor.is_best:
            dst = os.path.join(self.directory, "best")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    def on_fit_end(self, run):
        monitor = run.monitor
        if monitor.best_epoch != -1:
            self.log_metrics({"best_score": monitor.best_score}, monitor.best_epoch)
        self.client.log_artifacts(self.run_id, self.directory)
        self.client.set_terminated(self.run_id)
        shutil.rmtree(self.directory)

    def log_params(self, params):
        params = [Param(key, str(value)) for key, value in params.items()]
        self.client.log_batch(self.run_id, metrics=[], params=params, tags=[])

    def log_metrics(self, metrics, step=0):
        ts = int(time.time() * 1000)  # timestamp in milliseconds.
        metrics = [Metric(key, value, ts, step) for key, value in metrics.items()]
        self.client.log_batch(self.run_id, metrics=metrics, params=[], tags=[])
