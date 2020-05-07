import os
import re
import subprocess
from typing import Any, Dict, Iterator

from ivory import utils
from ivory.core import default, instance
from ivory.core.base import Base, Experiment
from ivory.core.evaluator import Evaluator
from ivory.core.run import Run


class Client(Base):
    """The Ivory client class."""

    def create_experiment(self, name: str) -> Experiment:
        """Creates an `Experiment` according to the YAML file specified by `name`.

        Args:
            path: experiment name.

        Returns:
            an experiment instance.
        """
        basename = name.split(".")[0]
        params, source_name = utils.path.load_params(basename, self.source_name)
        if "run" not in params:
            params = {"run": params}
        if "experiment" not in params:
            params.update(default.get("experiment"))
        if "name" not in params["experiment"]:
            params["experiment"]["name"] = name
        experiment = instance.create_base_instance(params, "experiment", source_name)
        if self.tracker:
            experiment.set_tracker(self.tracker)
        if self.tuner:
            experiment.set(tuner=self.tuner)
        return experiment

    def create_run(self, name: str, args=None, **kwargs):
        return self.create_experiment(name).create_run(args, **kwargs)

    def create_task(self, name: str):
        return self.create_experiment(name).create_task()

    def create_study(self, name: str):
        return self.create_experiment(name).create_study()

    def create_evaluator(self) -> Evaluator:
        return Evaluator(self)

    def search_run_ids(
        self,
        name: str = "",
        parent_run_id: str = "",
        parent_only: bool = False,
        nested_only: bool = False,
        exclude_parent: bool = False,
        **query,
    ) -> Iterator[str]:
        """Yields matching run ids.

        Args:
            name: experiment name pattern for filtering.
            parent_run_id: if specified, search from runs which have the parent id.
            parent_only: if True, search from parent runs.
            nested_only: if True, search from nested runs.
            exclude_parent: if True, skip parent runs.
            **query: key-value pairs for filtering.

        Yields:
            run_id
        """
        for experiment in self.tracker.list_experiments():
            if name and not re.match(name, experiment.name):
                continue
            yield from self.tracker.search_run_ids(
                experiment.experiment_id,
                parent_run_id,
                parent_only,
                nested_only,
                exclude_parent,
                **query,
            )

    def search_parent_run_ids(self, name: str = "", **query) -> Iterator[str]:
        return self.search_run_ids(name, parent_only=True, **query)

    def search_nested_run_ids(self, name: str = "", **query) -> Iterator[str]:
        return self.search_run_ids(name, nested_only=True, **query)

    def get_run_id(self, name: str, run_name: str, number: int) -> str:
        experiment_id = self.tracker.get_experiment_id(name)
        return self.tracker.get_run_id(experiment_id, run_name, number)

    def set_parent_run_id(self, run_id: str, parent_run_id: str):
        self.tracker.set_parent_run_id(run_id, parent_run_id)

    def get_parent_run_id(self, run_id: str) -> str:
        return self.tracker.get_parent_run_id(run_id)

    def set_terminated(self, name: str = ""):
        for run_id in self.search_run_ids(name):
            self.tracker.client.set_terminated(run_id)

    def load_params(self, run_id: str) -> Dict[str, Any]:
        return self.tracker.load_params(run_id)

    def load_run(self, run_id: str, mode: str = "test") -> Run:
        return self.tracker.load_run(run_id, mode)

    def load_instance(self, run_id: str, instance_name: str, mode: str = "test") -> Any:
        return self.tracker.load_instance(run_id, instance_name, mode)

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        try:
            subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])
        except KeyboardInterrupt:
            pass

    def update_params(self, name: str = "", **default):
        for experiment in self.tracker.list_experiments():
            if name and not re.match(name, experiment.name):
                continue
            self.tracker.update_params(experiment.experiment_id, **default)

    def remove_deleted_runs(self, name: str = "") -> int:
        """Remove deleted runs from local file system.

        Args:
            name: experiment name pattern for filtering.

        Returns:
            number of removed runs.
        """
        num_runs = 0
        for experiment in self.tracker.list_experiments():
            if name and not re.match(name, experiment.name):
                continue
            num_runs += self.tracker.remove_deleted_runs(experiment.experiment_id)
        return num_runs


def create_client(path="client", directory=".", tracker=True) -> Client:
    source_name = utils.path.normpath(path, directory)
    if os.path.exists(source_name):
        params, _ = utils.path.load_params(source_name)
    else:
        params = default.get("client")
    if not tracker and "tracker" in params["client"]:
        params["client"].pop("tracker")
    with utils.path.chdir(source_name):
        client = instance.create_base_instance(params, "client", source_name)
    return client
