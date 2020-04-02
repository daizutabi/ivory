import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlflow

import ivory.core.state
from ivory import utils
from ivory.callbacks.tracking import Tracking
from ivory.utils.mlflow import get_source_name, get_tags


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

    def list_run_ids(self, experiment_id: str):
        for run_info in self.client.list_run_infos(experiment_id):
            yield run_info.run_id

    def get_source_name(self, run_id):
        return get_source_name(self.client.get_run(run_id))

    def load_params(self, run_id):
        return load(self, run_id, "params")

    def load_run(self, run_id, mode, create_run):
        return load(self, run_id, "run", mode, create_run)

    def load_instance(self, run_id, name, mode, create_run, create_instance):
        return load(self, run_id, name, mode, create_run, create_instance)

    def create_tracking(self):
        return Tracking(self.tracking_uri)

    def update_params(self, experiment_id, **default):
        runs = []
        for run_id in self.list_run_ids(experiment_id):
            runs.append(self.client.get_run(run_id))
        args = []
        for run in runs:
            args.extend(list(run.data.params.keys()))
        args = list(set(args))
        tracking = self.create_tracking()
        for run in runs:
            run_id = run.info.run_id
            params = self.load_params(run_id)
            update = {}
            for arg in args:
                value = utils.get_value(params["run"], arg)
                if value is not None:
                    update[arg] = value
                elif arg in default:
                    update[arg] = default[arg]
            tracking.log_params(run_id, update)


params_cache: Dict[str, Dict[str, Any]] = {}


def load(tracker, run_id, name, mode=None, create_run=None, create_instance=None):
    if name == "params" and run_id in params_cache:
        return params_cache[run_id]
    source_name = tracker.get_source_name(run_id)
    client = tracker.client
    with utils.chdir(source_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            if run_id not in params_cache:
                params_path = client.download_artifacts(run_id, "params.yaml", tmpdir)
                params, _ = utils.load_params(params_path)
                params_cache[run_id] = params
                if name == "params":
                    return params
            params = params_cache[run_id]
            mode = get_valid_mode(client, run_id, mode)
            if name == "run":
                state_dict_path = client.download_artifacts(run_id, mode, tmpdir)
                run = create_run(params)
                state_dict = run.load(state_dict_path)
                run.load_state_dict(state_dict)
                return run
            os.mkdir(os.path.join(tmpdir, mode))
            path = os.path.join(mode, name)
            state_dict_path = client.download_artifacts(run_id, path, tmpdir)
            instance = create_instance(name, params)
            if isinstance(instance, ivory.core.state.State):
                state_dict = ivory.core.state.load(state_dict_path)
            else:
                run = create_run(params)
                state_dict = run.load_instance(state_dict_path)
            instance.load_state_dict(state_dict)
            return instance


def get_valid_mode(client, run_id, mode):
    modes = []
    for artifact in client.list_artifacts(run_id):
        if artifact.is_dir:
            modes.append(artifact.path)
    if mode == "test" and mode not in modes:
        mode = "best"
    if mode == "best" and mode not in modes:
        mode = "current"
    if mode == "current" and mode not in modes:
        raise ValueError(f"'{mode}' artifacts not found.")
    return mode
