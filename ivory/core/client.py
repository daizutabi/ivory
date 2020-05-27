"""
This module provides the Ivory Client class which is one of the main classes of
Ivory library.
"""
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import ivory.callbacks.results
from ivory import utils
from ivory.callbacks.results import Results
from ivory.core import default, instance
from ivory.core.base import Base, Experiment
from ivory.core.run import Run, Study, Task
from ivory.utils.tqdm import tqdm


class Client(Base):
    """The Ivory client class.

    Attributes:
        tracker (Tracker, optional): A Tracker instance for tracking run process.
        tuner (Tuner, optional): A Tuner instance for hyperparameter tuning.
    """

    def create_experiment(self, name: str) -> Experiment:
        """Creates an `Experiment` according to the YAML file specified by `name`.

        Args:
            name: Experiment name.
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
        return experiment

    def create_run(self, name: str, args=None, **kwargs) -> Run:
        """Creates a `Run`.

        Args:
            name: Experiment name.
            args (dict, optional): Parameter dictionary to update the default values
                of `Experiment`.
            **kwargs: Additional parameters.
        """
        return self.create_experiment(name).create_run(args, **kwargs)

    def create_task(self, name: str, run_number: Optional[int] = None) -> Task:
        """Creates a `Task`.

        Args:
            name: Experiment name.
            run_number: If specified, load an existing task instead of creating a new
                one.
        """
        if run_number is None:
            return self.create_experiment(name).create_task()
        else:
            return self.load_run_by_name(name, task=run_number)  # type:ignore

    def create_study(
        self, name: str, args=None, run_number: Optional[int] = None, **suggests
    ) -> Study:
        """Creates a `Study`.

        Args:
            name: Experiment name.
            run_number: If specified, load an existing study instead of creating a new
                one.
        """
        if run_number is None:
            study = self.create_experiment(name).create_study(args, **suggests)
        else:
            study = self.load_run_by_name(name, study=run_number)
        if self.tuner and "storage" not in study.params["study"]["tuner"]:
            study.set(tuner=self.tuner)
        return study

    def get_run_id(self, name: str, **kwargs) -> str:
        """Returns a RunID.

        Args:
            name: Experiment name.
            **kwargs: Run name. If you want to get a run with name of 'run#5',
                **kwargs shoud be `run=5`.
        """
        run_name = list(kwargs)[0]
        run_number = kwargs[run_name]
        if run_number == -1:
            return next(self.search_run_ids(name, run_name))
        else:
            experiment_id = self.tracker.get_experiment_id(name)
            return self.tracker.get_run_id(experiment_id, run_name, run_number)

    def get_run_ids(self, name: str, **kwargs) -> Iterator[str]:
        for run_name, run_numbers in kwargs.items():
            if isinstance(run_numbers, int):
                run_numbers = [run_numbers]
            for run_number in run_numbers:
                yield self.get_run_id(name, **{run_name: run_number})

    def get_parent_run_id(self, name: str, **kwargs) -> str:
        run_id = self.get_run_id(name, **kwargs)
        return self.tracker.get_parent_run_id(run_id)

    def get_nested_run_ids(self, name: str, **kwargs) -> Iterator[str]:
        run_name = list(kwargs)[0]
        run_numbers = kwargs.pop(run_name)
        parent_run_ids = self.get_run_ids(name, **{run_name: run_numbers})
        yield from self.search_run_ids(name, parent_run_id=parent_run_ids, **kwargs)

    def set_parent_run_id(self, name: str, **kwargs):
        parent = {name: number for name, number in kwargs.items() if name != "run"}
        parent_run_id = self.get_run_id(name, **parent)
        for run_id in self.get_run_ids(name, run=kwargs["run"]):
            self.tracker.set_parent_run_id(run_id, parent_run_id)

    def get_run_name(self, run_id: str) -> str:
        return self.tracker.get_run_name(run_id)

    def get_run_name_tuple(self, run_id: str) -> Tuple[str, int]:
        return self.tracker.get_run_name_tuple(run_id)

    def search_run_ids(
        self,
        name: str = "",
        run_name: str = "",
        parent_run_id: Union[str, Iterable[str]] = "",
        parent_only: bool = False,
        nested_only: bool = False,
        exclude_parent: bool = False,
        best_score_limit: Optional[float] = None,
        **query,
    ) -> Iterator[str]:
        """Yields matching run id.

        Args:
            name: Experiment name pattern for filtering.
            run_name: Run name pattern for filtering.
            parent_run_id: If specified, search from runs which have the parent id.
            parent_only: If True, search from parent runs.
            nested_only: If True, search from nested runs.
            exclude_parent: If True, skip parent runs.
            best_score_limit: Yields runs with the best score better than this value.
            **query: Key-value pairs for filtering.

        Yields:
            run_id
        """
        for experiment in self.tracker.list_experiments():
            if name and not re.match(name, experiment.name):
                continue
            yield from self.tracker.search_run_ids(
                experiment.experiment_id,
                run_name,
                parent_run_id,
                parent_only,
                nested_only,
                exclude_parent,
                best_score_limit,
                **query,
            )

    def search_parent_run_ids(self, name: str = "", **query) -> Iterator[str]:
        yield from self.search_run_ids(name, parent_only=True, **query)

    def search_nested_run_ids(self, name: str = "", **query) -> Iterator[str]:
        yield from self.search_run_ids(name, nested_only=True, **query)

    def set_terminated(self, name: str, status: Optional[str] = None, **kwargs):
        for run_id in self.get_run_ids(name, **kwargs):
            self.tracker.client.set_terminated(run_id, status=status)

    def set_terminated_all(self, name: str = ""):
        for run_id in self.search_run_ids(name):
            self.tracker.client.set_terminated(run_id)

    def load_params(self, run_id: str) -> Dict[str, Any]:
        return self.tracker.load_params(run_id)

    def load_run(self, run_id: str, mode: str = "test") -> Run:
        return self.tracker.load_run(run_id, mode)

    def load_run_by_name(self, name: str, mode: str = "test", **kwargs) -> Run:
        run_id = self.get_run_id(name, **kwargs)
        return self.load_run(run_id, mode)

    def load_instance(self, run_id: str, instance_name: str, mode: str = "test") -> Any:
        return self.tracker.load_instance(run_id, instance_name, mode)

    def load_results(
        self,
        run_ids: Union[str, Iterable[str]],
        callback=None,
        reduction: str = "none",
        verbose: bool = True,
    ) -> Results:
        """Loads results from multiple runs and concatenates them.

        Args:
            run_ids: Multiple run ids to load.
            callback (callable): Callback function for each run. This function must take
                a `(index, output, target)` and return the same signature.
            verbose: If `True`, tqdm progress bar is displayed.

        Returns:
            A concatenated results instance.
        """
        if isinstance(run_ids, str):
            return self.load_instance(run_ids, "results")
        run_ids = list(run_ids)
        it = (self.load_instance(run_id, "results") for run_id in run_ids)
        if verbose:
            it = tqdm(it, total=len(run_ids), leave=False)
        return ivory.callbacks.results.concatenate(
            it, callback=callback, reduction=reduction
        )

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
        """Removes deleted runs from a local file system.

        Args:
            name: A regex pattern of experiment name for filtering.

        Returns:
            Number of removed runs.
        """
        num_runs = 0
        for experiment in self.tracker.list_experiments():
            if name and not re.match(name, experiment.name):
                continue
            num_runs += self.tracker.remove_deleted_runs(experiment.experiment_id)
        return num_runs


def create_client(
    directory: str = "", name: str = "client", tracker: bool = True
) -> Client:
    """Creates an Ivory client.

    Args:
        directory: A working directory. If a YAML file specified by the `name`
            parameter exists, the file is loaded to configure the client. In addition,
            this directory is automatically inserted to `sys.path`.
        name: A YAML config file name.
        tracker: If true, the client instance has a tracker.

    Returns:
        An created client.

    Note:
        If `tracker` is True (default value), a `mlruns` directory is made under the
        working directory by the MLFlow Tracking.
    """
    if directory:
        path = os.path.abspath(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
    source_name = utils.path.normpath(name, directory)
    if os.path.exists(source_name):
        params, _ = utils.path.load_params(source_name)
    else:
        params = default.get("client")
    if not tracker and "tracker" in params["client"]:
        params["client"].pop("tracker")
    with utils.path.chdir(source_name):
        client = instance.create_base_instance(params, "client", source_name)
    return client
