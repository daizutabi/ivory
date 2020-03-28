from dataclasses import dataclass
import tempfile
from typing import Optional

import mlflow

from ivory import utils
from ivory.callbacks.tracking import Tracking
from ivory.utils.mlflow import get_source_name, get_tags
import ivory.core.state


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

    def search_runs(self, experiment_id, params=None, tags=None, return_id=True):
        if params is None:
            params = {}
        if tags is None:
            tags = {}
        runs = self.client.search_runs(experiment_id, filter_string(params, tags))
        if return_id:
            return [run.info.run_id for run in runs]
        else:
            return runs

    def get_source_name(self, run_or_run_id):
        if isinstance(run_or_run_id, str):
            run = self.client.get_run(run_or_run_id)
        else:
            run = run_or_run_id
        return get_source_name(run)

    def create_tracking(self):
        return Tracking(self.tracking_uri)

    def load_run(self, run_id, mode, create_run):
        source_name = self.get_source_name(run_id)
        client = self.client
        with utils.chdir(source_name):
            mode = get_valid_mode(client, run_id, mode)
            with tempfile.TemporaryDirectory() as tmpdir:
                params_path = client.download_artifacts(run_id, "params.yaml", tmpdir)
                state_dict_path = client.download_artifacts(run_id, mode, tmpdir)
                params = utils.load_params(params_path)
                run = create_run(params)
                state_dict = run.load(state_dict_path)
                run.load_state_dict(state_dict)
        return run

    def load_instance(self, run_id, name, mode, create_run, create_instance):
        source_name = self.get_source_name(run_id)
        client = self.client
        with utils.chdir(source_name):
            mode = get_valid_mode(client, run_id, mode)
            with tempfile.TemporaryDirectory() as tmpdir:
                params_path = client.download_artifacts(run_id, "params.yaml", tmpdir)
                state_dict_path = client.download_artifacts(run_id, mode, tmpdir)
                params = utils.load_params(params_path)
                instance = create_instance(name, params)
                if isinstance(instance, ivory.core.state.State):
                    state_dict = ivory.core.state.load(state_dict_path, name)
                else:
                    run = create_run(params)
                    state_dict = run.load_instance(state_dict_path, name)
                instance.load_state_dict(state_dict)
        return instance


def get_valid_mode(client, run_id, mode):
    modes = get_modes(client, run_id)
    if mode == "test" and mode not in modes:
        mode = "best"
    if mode == "best" and mode not in modes:
        mode = "current"
    if mode == "current" and mode not in modes:
        raise ValueError(f"'{mode}' artifacts not found.")
    return mode


def get_modes(client, run_id):
    modes = []
    for artifact in client.list_artifacts(run_id):
        if artifact.is_dir:
            modes.append(artifact.path)
    return modes


def filter_string(params, tags=None):
    """
    Examples:
        >>> params = {"lr": 1e-3, "fold": 2}
        >>> tags = {"mode": 'train'}
        >>> filter_string(params, tags)
        "param.lr='0.001' and param.fold='2' and tag.mode='train'"
    """
    filters = []
    for key, value in params.items():
        filters.append(f"param.{key}='{value}'")
    if tags:
        for key, value in tags.items():
            filters.append(f"tag.{key}='{value}'")
    return " and ".join(filters)
