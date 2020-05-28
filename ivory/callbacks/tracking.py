import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict

import mlflow
import yaml
from mlflow.entities import Metric, Param
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from ivory import utils
from ivory.core.run import Run


@dataclass
class Tracking:
    tracking_uri: str

    def __post_init__(self):
        self.client = mlflow.tracking.MlflowClient(self.tracking_uri)

    def on_epoch_end(self, run: Run):
        self.save_run(run, "current")
        if not run.metrics:
            return
        metrics = run.metrics.copy()
        monitor = run.monitor
        if monitor:
            metrics.update(best_score=monitor.best_score, best_epoch=monitor.best_epoch)
        self.log_metrics(run.id, metrics, run.metrics.epoch)

    def on_fit_end(self, run: Run):
        if self.client.get_run(run.id).info.status == "RUNNING":
            self.client.set_terminated(run.id)

    def on_test_end(self, run: Run):
        self.save_run(run, "test")

    def log_params_artifact(self, run: Run):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "params.yaml")
            with open(path, "w") as file:
                yaml.dump(run.params, file, sort_keys=False)
            with utils.path.chdir(run.source_name):
                self.client.log_artifacts(run.id, tmpdir)

    def log_params(self, run_id: str, params: Dict[str, Any]):
        params_list = []
        for key, value in params.items():
            if key != 'verbose':
                params_list.append(Param(key, to_str(value)))
        self.client.log_batch(run_id, metrics=[], params=params_list, tags=[])

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: int = 0):
        ts = int(time.time() * 1000)  # timestamp in milliseconds.
        metrics_ = [Metric(key, value, ts, step) for key, value in metrics.items()]
        self.client.log_batch(run_id, metrics=metrics_, params=[], tags=[])

    def save_run(self, run: Run, mode: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = os.path.join(tmpdir, mode)
            os.mkdir(directory)
            run.save(directory)
            with utils.path.chdir(run.source_name):
                self.client.log_artifacts(run.id, tmpdir)
                if mode != "current":
                    return
                if run.monitor and run.monitor.is_best and run.monitor.best_epoch > -1:
                    os.rename(directory, directory.replace("current", "best"))
                    self.client.log_artifacts(run.id, tmpdir)

    def set_tags(self, run_id: str, tags: Dict[str, Any]):
        for key, value in tags.items():
            if key != 'verbose':
                self.client.set_tag(run_id, key, to_str(value))

    def set_parent_run_id(self, run_id: str, parent_run_id: str):
        self.client.set_tag(run_id, MLFLOW_PARENT_RUN_ID, parent_run_id)


def to_str(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(to_str(x) for x in value) + "]"
    elif isinstance(value, float):
        return f"{value:.4g}"
    else:
        return str(value)
