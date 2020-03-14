import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import mlflow
import yaml
from mlflow.entities import Metric, Param
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from ivory.callbacks import Callback


@dataclass
class Tracking(Callback):
    experiment_id: str = ""
    tracking_uri: Optional[str] = None

    def on_fit_start(self, run):
        self.client = mlflow.tracking.MlflowClient(self.tracking_uri)
        tags = context_registry.resolve_tags({MLFLOW_RUN_NAME: run.name})
        tracking_run = self.client.create_run(self.experiment_id, tags=tags)
        run.run_id = tracking_run.info.run_id
        self.log_params(run.run_id, run.params)
        self.tmpdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.tmpdir, "current"))
        path = os.path.join(self.tmpdir, "params.yaml")
        with open(path, "w") as file:
            yaml.dump(run.params, file, sort_keys=False)

    def on_epoch_end(self, run):
        self.log_metrics(run.run_id, dict(run.metrics.record), run.metrics.epoch)
        src = os.path.join(self.tmpdir, "current")
        run.save(src)
        if run.monitor.is_best:
            dst = os.path.join(self.tmpdir, "best")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    def on_fit_end(self, run):
        monitor = run.monitor
        if monitor.best_epoch != -1:
            self.log_metrics({"best_score": monitor.best_score}, monitor.best_epoch)
        self.client.log_artifacts(run.run_id, self.tmpdir)
        self.client.set_terminated(run.run_id)
        shutil.rmtree(self.tmpdir)

    def log_params(self, run_id, params):
        params_list = []
        for key, value in params.items():
            if key == "experiment":
                continue
            value = str(value)[:250]
            params_list.append(Param(key, value))
        self.client.log_batch(run_id, metrics=[], params=params_list, tags=[])

    def log_metrics(self, run_id, metrics, step=0):
        ts = int(time.time() * 1000)  # timestamp in milliseconds.
        metrics = [Metric(key, value, ts, step) for key, value in metrics.items()]
        self.client.log_batch(run_id, metrics=metrics, params=[], tags=[])
