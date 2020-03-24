from dataclasses import dataclass
from typing import Optional

import mlflow

from ivory import utils
from ivory.callbacks.tracking import Tracking
from ivory.utils.mlflow import get_tags


@dataclass
class Tracker:
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None

    def __post_init__(self):
        self.client = mlflow.tracking.MlflowClient(self.tracking_uri)
        self.tracking_uri = self.client._tracking_client.tracking_uri
        if self.artifact_location:
            self.artifact_location = utils.to_uri(self.artifact_location)

    def get_experiment_id(self, name: str):
        experiment = self.client.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id

    def create_experiment(self, name: str):
        experiment_id = self.get_experiment_id(name)
        if not experiment_id:
            experiment_id = self.client.create_experiment(name, self.artifact_location)
        return experiment_id

    def create_run(self, experiment_id: str, name: str, source_name: str = ""):
        tags = get_tags(name, source_name)
        run = self.client.create_run(experiment_id, tags=tags)
        return run.info.run_id

    def list_run_infos(self, experiment_id):
        return self.client.list_run_infos(experiment_id)

    def search_runs(self, experiment_id, params=None, tags=None):
        if params is None:
            params = {}
        if tags is None:
            tags = {}
        filter_string = utils.filter_string(params, tags)
        runs = self.client.search_runs(experiment_id, filter_string)
        return [run.info.run_id for run in runs]

    def create_tracking(self, experiment_id, param_names=None):
        return Tracking(experiment_id, self.tracking_uri, param_names)
