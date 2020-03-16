from dataclasses import dataclass
from typing import Optional

import mlflow

from ivory import utils
from ivory.callbacks.tracking import Tracking
from ivory.core import instance


@dataclass
class Tracker:
    name: str = ""
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None

    def create_experiment(self, name: str):
        """Creates an tracking experiment and returns its id."""
        self.name = name
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            if self.artifact_location:
                self.artifact_location = utils.to_uri(self.artifact_location)
            experiment_id = client.create_experiment(name, self.artifact_location)
            experiment = client.get_experiment(experiment_id)
        self.artifact_location = experiment.artifact_location
        return experiment_id

    def get_run_id(self, experiment_id, run_index):
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        run_infos = client.list_run_infos(experiment_id)
        return run_infos[run_index].run_id

    def create_tracking(self, experiment_id):
        return Tracking(experiment_id, self.tracking_uri)


def create_tracker(params):
    """Creates an `Tracker` instance."""
    if isinstance(params, str):
        params = utils.load_params(params)
    if "tracker" in params:
        return instance.instantiate(params["tracker"])
    else:
        return instance.instantiate(params["experiment"]["tracker"])
