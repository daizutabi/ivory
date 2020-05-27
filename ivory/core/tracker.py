import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import mlflow
from logzero import logger
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.utils.mlflow_tags import (MLFLOW_GIT_COMMIT, MLFLOW_PARENT_RUN_ID,
                                      MLFLOW_RUN_NAME, MLFLOW_SOURCE_NAME)

import ivory.core.state
from ivory import utils
from ivory.callbacks.tracking import Tracking
from ivory.core import instance
from ivory.core.run import Run


@dataclass
class Tracker:
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None

    def __post_init__(self):
        self.client = mlflow.tracking.MlflowClient(self.tracking_uri)
        self.tracking_uri = self.client._tracking_client.tracking_uri
        if self.artifact_location:
            self.artifact_location = utils.path.to_uri(self.artifact_location)

    def get_experiment_id(self, name: str):
        experiment = self.client.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id

    def get_run_id(self, experiment_id: str, name: str, number: int):
        run_name = create_run_name(name, number)
        for run in self.client.search_runs(experiment_id):
            if get_run_name(run) == run_name:
                return run.info.run_id

    def create_experiment(self, name: str) -> str:
        experiment_id = self.get_experiment_id(name)
        if not experiment_id:
            logger.info(f"A new experiment created with name: {name!r}")
            experiment_id = self.client.create_experiment(name, self.artifact_location)
        return experiment_id

    def get_run_number(self, experiment_id: str, name: str) -> int:
        name = create_run_name(name)
        number = -1
        for run in self.client.search_runs(experiment_id, run_view_type=3):
            run_name = get_run_name(run)
            if run_name.startswith(name):
                number = max(number, int(run_name.split("#")[1]))
        return number

    def create_run(
        self, experiment_id: str, name: str, source_name: str = ""
    ) -> Tuple[str, str]:
        number = self.get_run_number(experiment_id, name) + 1
        run_name = create_run_name(name, number)
        tags = create_tags(run_name, source_name)
        run = self.client.create_run(experiment_id, tags=tags)
        return run_name, run.info.run_id

    def create_tracking(self) -> Tracking:
        return Tracking(self.tracking_uri)  # type:ignore

    def list_experiments(self, view_type=None) -> List:
        return self.client.list_experiments(view_type)

    def list_run_ids(
        self,
        experiment_id: str,
        parent_run_id: Union[str, Iterable[str]] = "",
        exclude_parent: bool = False,
    ) -> Iterator[str]:
        if parent_run_id:
            yield from self.list_nested_run_ids(experiment_id, parent_run_id)
        else:
            if exclude_parent:
                parent_run_ids = list(self.list_parent_run_ids(experiment_id))
            for run_info in self.client.list_run_infos(experiment_id):
                run_id = run_info.run_id
                if not exclude_parent or run_id not in parent_run_ids:
                    yield run_id

    def list_nested_run_ids(
        self, experiment_id: str, parent_run_id: Union[str, Iterable[str]] = ""
    ) -> Iterator[str]:
        if not isinstance(parent_run_id, str):
            for parent_run_id in list(parent_run_id):
                yield from self.list_nested_run_ids(experiment_id, parent_run_id)
            return
        filter_string = ""
        if parent_run_id:
            filter_string = f"tags.{MLFLOW_PARENT_RUN_ID}={parent_run_id!r}"
        for run in self.client.search_runs(experiment_id, filter_string):
            if MLFLOW_PARENT_RUN_ID in run.data.tags:
                yield run.info.run_id

    def list_parent_run_ids(self, experiment_id: str) -> Iterator[str]:
        parent_run_ids: List[str] = []
        for run in self.client.search_runs(experiment_id):
            if MLFLOW_PARENT_RUN_ID in run.data.tags:
                parent_run_id = run.data.tags[MLFLOW_PARENT_RUN_ID]
                if parent_run_id not in parent_run_ids:
                    yield parent_run_id
                    parent_run_ids.append(parent_run_id)

    def remove_deleted_runs(self, experiment_id: str) -> int:
        experiment = self.client.get_experiment(experiment_id)
        uri = experiment.artifact_location
        if not uri.startswith("file"):
            raise NotImplementedError
        path = utils.path.local_file_uri_to_path(uri)
        num_runs = 0
        for run_info in self.client.list_run_infos(experiment_id, run_view_type=2):
            run_id = run_info.run_id
            directory = os.path.normpath(os.path.join(path, run_id))
            if os.path.exists(directory):
                shutil.rmtree(directory)
                num_runs += 1
        return num_runs

    def search_run_ids(
        self,
        experiment_id: str,
        run_name: str = "",
        parent_run_id: Union[str, Iterable[str]] = "",
        parent_only: bool = False,
        nested_only: bool = False,
        exclude_parent: bool = False,
        best_score_limit: Optional[float] = None,
        **query,
    ) -> Iterator[str]:
        if parent_only:
            run_ids = self.list_parent_run_ids(experiment_id)
        elif nested_only:
            run_ids = self.list_nested_run_ids(experiment_id)
        else:
            run_ids = self.list_run_ids(experiment_id, parent_run_id, exclude_parent)
        for run_id in run_ids:
            if run_name and not re.match(run_name, self.get_run_name(run_id)):
                continue
            if query:
                params = self.load_params(run_id)
                if not utils.params.match(params, **query):
                    continue
            if best_score_limit is not None:
                monitor = self.load_instance(run_id, "monitor", "test")
                if monitor.mode == "min" and monitor.best_score <= best_score_limit:
                    yield run_id
                elif monitor.mode == "max" and monitor.best_score >= best_score_limit:
                    yield run_id
            else:
                yield run_id

    def get_run_name(self, run_id: str) -> str:
        return get_run_name(self.client.get_run(run_id))

    def get_run_name_tuple(self, run_id: str) -> Tuple[str, int]:
        run_name = self.get_run_name(run_id)
        return split_run_name(run_name)

    def get_source_name(self, run_id: str) -> str:
        return get_source_name(self.client.get_run(run_id))

    def get_parent_run_id(self, run_id: str) -> str:
        return get_parent_run_id(self.client.get_run(run_id))

    def get_run_status(self, run_id: str) -> str:
        return self.client.get_run(run_id).info.status

    def load_params(self, run_id: str) -> Dict[str, Any]:
        return load(self, run_id, "params")

    def load_run(self, run_id: str, mode: str) -> Run:
        name, _ = self.get_run_name_tuple(run_id)
        run = load(self, run_id, name, mode=mode)
        run.source_name = self.get_source_name(run_id)
        run.set_tracker(self, name)
        return run

    def load_instance(self, run_id: str, instance_name: str, mode: str) -> Any:
        name, _ = self.get_run_name_tuple(run_id)
        return load(self, run_id, name, instance_name, mode=mode)

    def load_state_dict(self, run: Run, mode: str):
        mode = get_valid_mode(self.client, run.id, mode)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dict_path = self.client.download_artifacts(run.id, mode, tmpdir)
            state_dict = run.load(state_dict_path)
            run.load_state_dict(state_dict)

    def update_params(self, experiment_id: str, **default):
        runs = []
        for run_id in self.list_run_ids(experiment_id, exclude_parent=True):
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
                value = utils.params.get_value(params["run"], arg)
                if value is not None:
                    update[arg] = value
                elif arg in default:
                    update[arg] = default[arg]
            tracking.log_params(run_id, update)

    def set_parent_run_id(self, run_id: str, parent_run_id: str):
        self.client.set_tag(run_id, MLFLOW_PARENT_RUN_ID, parent_run_id)


params_cache: Dict[str, Dict[str, Any]] = {}


def load(
    tracker: Tracker, run_id: str, name: str, instance_name: str = "", mode: str = ""
):
    if name == "params" and run_id in params_cache:
        return params_cache[run_id]
    source_name = tracker.get_source_name(run_id)
    client = tracker.client
    with utils.path.chdir(source_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            if run_id not in params_cache:
                params_path = client.download_artifacts(run_id, "params.yaml", tmpdir)
                params, _ = utils.path.load_params(params_path)
                params_cache[run_id] = params
                if name == "params":
                    return params
            params = params_cache[run_id]
            mode = get_valid_mode(client, run_id, mode)
            if not instance_name:
                run = create_run(params, name)
                if mode:
                    state_dict_path = client.download_artifacts(run_id, mode, tmpdir)
                    state_dict = run.load(state_dict_path)
                    run.load_state_dict(state_dict)
                return run
            instance = create_instance(params, name, instance_name)
            if not mode:
                return instance
            os.mkdir(os.path.join(tmpdir, mode))
            path = os.path.join(mode, instance_name)
            state_dict_path = client.download_artifacts(run_id, path, tmpdir)
            if isinstance(instance, ivory.core.state.State):
                state_dict = ivory.core.state.load(state_dict_path)
            else:
                run = create_run(params, name)
                state_dict = run.load_instance(state_dict_path)
            instance.load_state_dict(state_dict)
            return instance


def create_run(params: Dict[str, Any], name: str) -> Run:
    return instance.create_base_instance(params, name)


def create_instance(params: Dict[str, Any], name: str, instance_name: str) -> Any:
    return instance.create_instance(params[name], instance_name)


def get_valid_mode(client: MlflowClient, run_id: str, mode: str) -> str:
    modes = []
    for artifact in client.list_artifacts(run_id):
        if artifact.is_dir:
            modes.append(artifact.path)
    if mode == "test" and mode not in modes:
        mode = "best"
    if mode == "best" and mode not in modes:
        mode = "current"
    if mode == "current" and mode not in modes:
        mode = ""
    return mode


git_cache: Dict[str, str] = {}


def create_run_name(name: str, number: int = -1) -> str:
    # name = name[0].upper() + name[1:]
    if number == -1:
        return name
    # return f"{name}#{number:03d}"
    return f"{name}#{number}"


def split_run_name(run_name: str) -> Tuple[str, int]:
    name, number = run_name.split("#")
    return name.lower(), int(number)


def create_tags(run_name: str, source_name: str = "") -> Dict[str, str]:
    tags = {MLFLOW_RUN_NAME: run_name}
    if source_name:
        tags[MLFLOW_SOURCE_NAME] = source_name
        if source_name not in git_cache:
            git_cache[source_name] = _get_git_commit(source_name)
        if git_cache[source_name]:
            tags[MLFLOW_GIT_COMMIT] = git_cache[source_name]
    tags = context_registry.resolve_tags(tags)
    return tags


def get_run_name(run) -> str:
    return run.data.tags[MLFLOW_RUN_NAME]


def get_source_name(run) -> str:
    return run.data.tags[MLFLOW_SOURCE_NAME]


def get_parent_run_id(run) -> str:
    return run.data.tags[MLFLOW_PARENT_RUN_ID]
