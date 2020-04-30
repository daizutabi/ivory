import os
import tempfile
import time
from dataclasses import dataclass

import mlflow
import yaml
from mlflow.entities import Metric, Param
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from ivory import utils


@dataclass
class Tracking:
    tracking_uri: str

    def __post_init__(self):
        self.client = mlflow.tracking.MlflowClient(self.tracking_uri)

    def on_epoch_end(self, run):
        metrics = run.metrics.copy()
        monitor = run.monitor
        if monitor:
            metrics.update(best_score=monitor.best_score, best_epoch=monitor.best_epoch)
        self.log_metrics(run.id, metrics, run.metrics.epoch)
        self.save_run(run, "current")

    def on_fit_end(self, run):
        self.set_terminated(run.id)

    def on_test_end(self, run):
        self.save_run(run, "test")
        self.set_terminated(run.id)

    def set_terminated(self, run_id):
        self.client.set_terminated(run_id)

    def save_run(self, run, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = os.path.join(tmpdir, mode)
            os.mkdir(directory)
            run.save(directory)
            with utils.chdir(run.source_name):
                self.client.log_artifacts(run.id, tmpdir)
                if mode != "current":
                    return
                if run.monitor and run.monitor.is_best:
                    os.rename(directory, directory.replace("current", "best"))
                    self.client.log_artifacts(run.id, tmpdir)

    def log_params_artifact(self, run):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "params.yaml")
            with open(path, "w") as file:
                yaml.dump(run.params, file, sort_keys=False)
            with utils.chdir(run.source_name):
                self.client.log_artifacts(run.id, tmpdir)

    def log_params(self, run_id, params):
        params_list = []
        for key, value in params.items():
            params_list.append(Param(key, to_str(value)))
        self.client.log_batch(run_id, metrics=[], params=params_list, tags=[])

    def log_metrics(self, run_id, metrics, step=0):
        ts = int(time.time() * 1000)  # timestamp in milliseconds.
        metrics = [Metric(key, value, ts, step) for key, value in metrics.items()]
        self.client.log_batch(run_id, metrics=metrics, params=[], tags=[])

    def set_tags(self, run_id, tags):
        for key, value in tags.items():
            self.client.set_tag(run_id, key, to_str(value))

    def set_parent_run_id(self, run_id, parent_run_id):
        self.client.set_tag(run_id, MLFLOW_PARENT_RUN_ID, parent_run_id)


def to_str(value):
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(to_str(x) for x in value) + "]"
    elif isinstance(value, float):
        return f"{value:.4g}"
    else:
        return str(value)
