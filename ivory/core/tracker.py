from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_SOURCE_NAME

from ivory import utils
from ivory.callbacks.tracking import Tracking
from ivory.core import instance


@dataclass
class Tracker:
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None

    def __post_init__(self):
        self.client = mlflow.tracking.MlflowClient(self.tracking_uri)
        self.tracking_uri = self.client._tracking_client.tracking_uri

    def create_experiment(self, name: str):
        experiment = self.client.get_experiment_by_name(name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            if self.artifact_location:
                self.artifact_location = utils.to_uri(self.artifact_location)
            experiment_id = self.client.create_experiment(name, self.artifact_location)
        return experiment_id

    def create_run(self, name: str, experiment_id: str, source_name=None):
        tags = {MLFLOW_RUN_NAME: name}
        if source_name:
            tags[MLFLOW_SOURCE_NAME] = source_name
        tags = context_registry.resolve_tags(tags)
        run = self.client.create_run(experiment_id, tags=tags)
        return run.info.run_id

    def list_run_infos(self, experiment_id):
        return self.client.list_run_infos(experiment_id)

    def create_tracking(self, experiment_id, param_names=None):
        return Tracking(experiment_id, self.tracking_uri, param_names)


def create_tracker(params="params.yaml"):
    if isinstance(params, str):
        params = utils.load_params(params)
    if "environment" in params:
        params = params["environment"]
    if "tracker" in params:
        params = params["tracker"]
    return instance.instantiate(params)
