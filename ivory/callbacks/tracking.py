import os
import tempfile
import time

import mlflow
import yaml
from mlflow.entities import Metric, Param

from ivory import utils


class Tracking:
    def __init__(self, tracking_uri):
        self.client = mlflow.tracking.MlflowClient(tracking_uri)

    def on_fit_start(self, run):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "params.yaml")
            with open(path, "w") as file:
                yaml.dump(run.params, file, sort_keys=False)
            with utils.chdir(run.source_name):
                self.client.log_artifacts(run.id, tmpdir)

    def on_epoch_end(self, run):
        metrics = run.metrics.copy()
        monitor = run.monitor
        if monitor and monitor.best_epoch > 0:
            metrics.update(best_score=monitor.best_score, best_epoch=monitor.best_epoch)
        self.log_metrics(run.id, metrics, run.metrics.epoch)
        self.save_run(run, "current")

    def on_fit_end(self, run):
        self.client.set_terminated(run.id)

    def on_test_end(self, run):
        self.save_run(run, "test")

    def save_run(self, run, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = os.path.join(tmpdir, mode)
            os.mkdir(directory)
            run.save(directory)
            with utils.chdir(run.source_name):
                self.client.log_artifacts(run.id, tmpdir)
                if mode != "current":
                    return
                monitor = run.monitor
                if monitor and monitor.is_best and monitor.best_epoch > 0:
                    os.rename(directory, directory.replace("current", "best"))
                    self.client.log_artifacts(run.id, tmpdir)

    def log_params(self, run_id, params):
        params_list = []
        for key, value in params.items():
            value = str(value)[:250]
            params_list.append(Param(key, value))
        self.client.log_batch(run_id, metrics=[], params=params_list, tags=[])

    def log_metrics(self, run_id, metrics, step=0):
        ts = int(time.time() * 1000)  # timestamp in milliseconds.
        metrics = [Metric(key, value, ts, step) for key, value in metrics.items()]
        self.client.log_batch(run_id, metrics=metrics, params=[], tags=[])

    def set_tags(self, run_id, tags):
        for key, value in tags.items():
            self.client.set_tag(run_id, key, value)
