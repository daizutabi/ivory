import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

import mlflow
import yaml
from mlflow.entities import Metric, Param
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from ivory.utils.params import dot_get


@dataclass
class Tracking:
    experiment_id: str = ""
    tracking_uri: Optional[str] = None
    param_names: Optional[List[str]] = None

    def on_fit_start(self, run):
        self.client = mlflow.tracking.MlflowClient(self.tracking_uri)
        tags = context_registry.resolve_tags({MLFLOW_RUN_NAME: run.name})
        if not run.id:
            tracking_run = self.client.create_run(self.experiment_id, tags=tags)
            run.id = tracking_run.info.run_id
            run.params["run"]["id"] = run.id
        if self.param_names:
            params = get_params(run.params["run"], self.param_names)
            self.log_params(run.id, params)
        self.tmpdir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.tmpdir, "current"))
        path = os.path.join(self.tmpdir, "params.yaml")
        with open(path, "w") as file:
            yaml.dump(run.params, file, sort_keys=False)

    def on_epoch_end(self, run):
        self.log_metrics(run.id, run.metrics.record, run.metrics.epoch)
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
            best_score = {"best_score": monitor.best_score}
            self.log_metrics(run.id, best_score, monitor.best_epoch)
        self.client.log_artifacts(run.id, self.tmpdir)
        self.client.set_terminated(run.id)
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


def get_params(params, param_names):
    params_dict = {}
    for name in param_names:
        value = dot_get(params, name)
        if value is not None:
            name = name.split(".")[-1]
            params_dict[name] = value
    return params_dict
