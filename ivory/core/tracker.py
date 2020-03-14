from dataclasses import dataclass
from typing import Optional

import mlflow

from ivory import utils
from ivory.callbacks.tracking import Tracking


@dataclass
class Tracker:
    name: str = ""
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None

    def create_experiment(self, name: str):
        """Creates an tracking experiment and returns its id."""
        self.name = name
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        if self.artifact_location:
            self.artifact_location = utils.to_uri(self.artifact_location)
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            id_ = experiment.experiment_id
        else:
            id_ = client.create_experiment(name, self.artifact_location)
            experiment = client.get_experiment(id_)
        self.tracking_uri = client._tracking_client.tracking_uri
        self.artifact_location = experiment.artifact_location
        return id_

    def create_callback(self, experiment_id):
        return Tracking(experiment_id, self.tracking_uri)
