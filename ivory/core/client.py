import os
import re
import subprocess
from typing import Any, Dict, Iterable, Iterator, Tuple

import ivory.utils.data
from ivory import utils
from ivory.core import default, instance
from ivory.core.base import Base
from ivory.core.experiment import Experiment
from ivory.core.run import Run
from ivory.utils.tqdm import tqdm


class Client(Base):
    """The Ivory client class."""

    def create_experiment(self, path: str, name: str = "") -> Experiment:
        """Creates an `Experiment` according to the YAML file specified by `path`.

        By default, the experiment name is equal to the `path`, but if `name` is given,
        the name is obtained by "`path`.`name`".

        Args:
            path: file path without extension.
            name: suffix name for the experiment.

        Returns:
            an experiment instance.
        """
        params, source_name = utils.load_params(path, self.source_name)
        if "experiment" not in params:
            params.update(default.get("experiment"))
        if "name" not in params["experiment"]:
            if name:
                path = ".".join([path, name])
            params["experiment"]["name"] = path
        experiment = instance.create_base_instance(params, "experiment", source_name)
        if self.tracker:
            experiment.set_tracker(self.tracker)
        return experiment

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

    def search_parent_run_ids(self, name: str = "", **query) -> Iterator[str]:
        return self.search_run_ids(name, parent_only=True, **query)

    def search_nested_run_ids(self, name: str = "", **query) -> Iterator[str]:
        return self.search_run_ids(name, nested_only=True, **query)

    def get_parent_run_id(self, run_id: str) -> str:
        return self.tracker.get_parent_run_id(run_id)

    def update_params(self, name: str = "", **default):
        for experiment in self.tracker.list_experiments():
            if name and not re.match(name, experiment.name):
                continue
            self.tracker.update_params(experiment.experiment_id, **default)

    def set_terminated(self, name: str = ""):
        for run_id in self.search_run_ids(name):
            self.tracker.client.set_terminated(run_id)

    def load_params(self, run_id: str) -> Dict[str, Any]:
        return self.tracker.load_params(run_id)

    def load_run(self, run_id: str, mode: str = "test") -> Run:
        run = self.tracker.load_run(run_id, mode)
        run.set_tracker(self.tracker)
        return run

    def load_instance(self, run_id: str, instance_name: str, mode: str = "test") -> Any:
        return self.tracker.load_instance(run_id, instance_name, mode)

    def load_results(self, run_ids: Iterable[str], verbose: bool = True) -> Tuple:
        if verbose:
            run_ids = tqdm(list(run_ids), leave=False)
        it = (self.load_instance(run_id, "results", "test") for run_id in run_ids)
        return ivory.utils.data.concat_results(it)

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        try:
            subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])
        except KeyboardInterrupt:
            pass


def create_client(path="client", directory=".", tracker=True) -> Client:
    source_name = utils.normpath(path, directory)
    if os.path.exists(source_name):
        params, _ = utils.load_params(source_name)
    else:
        params = default.get("client")
    if not tracker and "tracker" in params["client"]:
        params["client"].pop("tracker")
    with utils.chdir(source_name):
        client = instance.create_base_instance(params, "client", source_name)
    return client
