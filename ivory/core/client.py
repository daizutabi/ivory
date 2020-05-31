"""
This module provides the Ivory Client class which is one of the main classes of
Ivory library.

To create an `Client` instance:

    import ivory

    client = ivory.create_client()

Here, the current directory becomes the working directory in which experiment
YAML files exist. If you want to refer other directory, use:

    client = ivory.create_client('path/to/working_directory')
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
    """The Ivory Client class.

    Attributes:
        tracker (Tracker): A Tracker instance for tracking run process.
        tuner (Tuner): A Tuner instance for hyperparameter tuning.
    """

    def __init__(self, params=None, **objects):
        super().__init__(params, **objects)
        self.experiments: Dict[str, Experiment] = {}

    def create_experiment(self, name: str, *args, **kwargs) -> Experiment:
        """Creates an `Experiment` according to the YAML file specified by `name`.

        Args:
            name: Experiment name.
            *args: Additional parameter files.
            **kwargs: Additional parameter files.

        A YAML file named `<name>.yml` or `<name>.yaml` should exist under the
        working directory.

        Any additionanl parameter files are added through `*args` and/or `**kwargs`.

        Examples:
            **Positional argument style**:

                experiment = client.create_experiment('example', 'study')

            In this case, `study.yml` is like this, including the instance name `study`:

                study:
                  tuner:
                    pruner:
                    class: optuna.pruners.MedianPruner
                  objective:
                    lr: example.suggest_lr

            **Keyword argument style**:

                experiment = client.create_experiment('example', study='study')

            In this case, `study.yml` is like this, omitting the instance name `study`:

                tuner:
                  pruner:
                    class: optuna.pruners.MedianPruner
                objective:
                  lr: example.suggest_lr
        """
        if name in self.experiments and not args and not kwargs:
            return self.experiments[name]

        params, source_name = utils.path.load_params(name, self.source_name)
        if "run" not in params:
            params = {"run": params}
        if "experiment" not in params:
            params.update(default.get("experiment"))
        if "name" not in params["experiment"]:
            params["experiment"]["name"] = name
        for value in args:
            option, _ = utils.path.load_params(value, self.source_name)
            params.update(option)
        for key, value in kwargs.items():
            option, _ = utils.path.load_params(value, self.source_name)
            if key not in option:
                option = {key: option}
            params.update(option)
        experiment = instance.create_base_instance(params, "experiment", source_name)
        if self.tracker:
            experiment.set_tracker(self.tracker)
        self.experiments[name] = experiment
        return experiment

    def create_run(self, name: str, args=None, **kwargs) -> Run:
        """Creates a `Run`.

        Args:
            name: Experiment name.
            args (dict, optional): Parameter dictionary to update the default values
                of `Experiment`.
            **kwargs: Additional parameters.

        Examples:
            To update a fold number:

                run = client.create_run('example', fold=3)

            If a parameter name includes dots:

                run = client.create_run('example', {'model.class': 'your.new.Model'})
        """
        return self.create_experiment(name).create_run(args, **kwargs)

    def create_task(self, name: str, run_number: Optional[int] = None) -> Task:
        """Creates a `Task` instance for multiple runs.

        Args:
            name: Experiment name.
            run_number (int, optional): If specified, load an existing task instead of
                creating a new one.

        See Also:
            [Multiple Runs](/tutorial/task) in Tutorial
        """
        if run_number is None:
            return self.create_experiment(name).create_task()
        else:
            return self.load_run_by_name(name, task=run_number)  # type:ignore

    def create_study(
        self, name: str, args=None, run_number: Optional[int] = None, **suggests
    ) -> Study:
        """Creates a `Study` instance for hyperparameter tuning.

        Args:
            name: Experiment name.
            args (str or dict): Suggest name (str) or parametric optimization (dict).
            run_number (int, optional): If specified, load an existing study instead of
                creating a new one.
            **suggests: Parametric optimization.

        Examples:
            To use a suggest function:

                study = client.create_study('example', 'lr')

            For parametric optimization:

                study = client.create_study('example', lr=(1e-5, 1e-3))

            If a parameter name includes dots:

                study = client.create_study('example', {'hidden_sizes.0': range(5, 20)})

        See Also:
            [Hyperparameter Tuning](/tutorial/tuning) in Tutorial
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

        Examples:
            To get a RunID of run#4.

                client.get_run_id('example', run=4)

            To get a RunID of task#10.

                client.get_run_id('example', task=10)
        """
        run_name = list(kwargs)[0]
        run_number = kwargs[run_name]
        if run_number == -1:
            return next(self.search_run_ids(name, run_name))
        else:
            experiment_id = self.tracker.get_experiment_id(name)
            return self.tracker.get_run_id(experiment_id, run_name, run_number)

    def get_run_ids(self, name: str, **kwargs) -> Iterator[str]:
        """Returns an iterator that yields RunIDs.

        Args:
            name: Experiment name.

        Examples:
            To get an iterator that yields RunIDs for Runs.

                client.get_run_id('example', run=[1, 2, 3])

            To get an iterator that yields RunIDs for Tasks.

                client.get_run_id('example', task=range(3, 8))
        """
        for run_name, run_numbers in kwargs.items():
            if isinstance(run_numbers, int):
                run_numbers = [run_numbers]
            for run_number in run_numbers:
                yield self.get_run_id(name, **{run_name: run_number})

    def get_parent_run_id(self, name: str, **kwargs) -> str:
        """Returns a parent RunID of a nested run.

        Args:
            name: Experiment name.

        Examples:
            To get a prarent RunID of run#5.

                client.get_parent_run_id('example', run=5)
        """
        run_id = self.get_run_id(name, **kwargs)
        return self.tracker.get_parent_run_id(run_id)

    def get_nested_run_ids(self, name: str, **kwargs) -> Iterator[str]:
        """Returns an iterator that yields nested RunIDs of parent runs.

        Args:
            name: Experiment name.

        Examples:
            To get an iterator that yields RunIDs of runs whose parent is task#2.

                client.get_nested_run_ids('example', task=2)

            Multiple parents can be specified.

                client.get_nested_run_ids('example', task=range(3, 8))
        """
        run_name = list(kwargs)[0]
        run_numbers = kwargs.pop(run_name)
        parent_run_ids = self.get_run_ids(name, **{run_name: run_numbers})
        yield from self.search_run_ids(name, parent_run_id=parent_run_ids, **kwargs)

    def set_parent_run_id(self, name: str, **kwargs):
        """Sets parent RunID to runs.

        Args:
            name: Experiment name.

        Examples:
            To set task#2 as a parant for run#4.

                client.set_parent_run_id('example', task=2, run=4)

            Multiple nested runs can be specified.

                client.set_parent_run_id('example', task=2, run=range(3))
        """
        parent = {name: number for name, number in kwargs.items() if name != "run"}
        parent_run_id = self.get_run_id(name, **parent)
        for run_id in self.get_run_ids(name, run=kwargs["run"]):
            self.tracker.set_parent_run_id(run_id, parent_run_id)

    def get_run_name(self, run_id: str) -> str:
        """Returns a run name (`run#XXX`, `task#XXX`, *etc*.) for RunID.

        Args:
            run_id: RunID
        """
        return self.tracker.get_run_name(run_id)

    def get_run_name_tuple(self, run_id: str) -> Tuple[str, int]:
        """Returns a run name as a tuple of (run class name, run number).

        Args:
            run_id: RunID
        """
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
        """Returns an iterator that yields matching RunIDs.

        Args:
            name: Experiment name pattern for filtering.
            run_name: Run name pattern for filtering.
            parent_run_id (str or iterable of str): If specified, search from runs
                which have the parent id(s).
            parent_only: If True, search from parent runs.
            nested_only: If True, search from nested runs.
            exclude_parent: If True, skip parent runs.
            best_score_limit: Yields runs with the best score better than this value.
            **query: Key-value pairs for filtering.
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
        """Returns an iterator that yields matching parent RunIDs.

        Args:
            name: Experiment name pattern for filtering.
            **query: Key-value pairs for filtering.
        """
        yield from self.search_run_ids(name, parent_only=True, **query)

    def search_nested_run_ids(self, name: str = "", **query) -> Iterator[str]:
        """Returns an iterator that yields matching nested RunIDs.

        Args:
            name: Experiment name pattern for filtering.
            **query: Key-value pairs for filtering.
        """
        yield from self.search_run_ids(name, nested_only=True, **query)

    def set_terminated(self, name: str, status: Optional[str] = None, **kwargs):
        """Sets runs' status to terminated.

        Args:
            status: A string value of
                [`mlflow.entities.RunStatus`](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunStatus).
                Defaults to “FINISHED”.

        Examples:
            To terminate a run:

                client.set_terminated('example', run=5)

            To kill multiple runs:

                client.set_terminated('example', 'KILLED', run=[3, 5, 7])
        """
        for run_id in self.get_run_ids(name, **kwargs):
            self.tracker.client.set_terminated(run_id, status=status)

    def set_terminated_all(self, name: str = ""):
        """Sets all runs' status to terminated.

        Args:
            status: A string value of
                [`mlflow.entities.RunStatus`](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunStatus).
                Defaults to “FINISHED”.

        Examples:
            To terminate all of the runs of the `example` experiment:

                client.set_terminated_all('example')

            To terminate all of the runs globally:

                client.set_terminated_all()
        """
        for run_id in self.search_run_ids(name):
            self.tracker.client.set_terminated(run_id)

    def load_params(self, run_id: str) -> Dict[str, Any]:
        """Returns a parameter dictionary loaded from MLFlow Tracking.

        Args:
            run_id: RunID for a run to be loaded.
        """
        return self.tracker.load_params(run_id)

    def load_run(self, run_id: str, mode: str = "test") -> Run:
        """Returns a `Run` instance created using parameters loaded from MLFlow
        Tracking.

        Args:
            run_id: RunID for a run to be loaded.
            mode: Mode name: `'current'`, `'best'`, or `'test'`.
                Default is `'{default}'`.
        """
        return self.tracker.load_run(run_id, mode)

    def load_run_by_name(self, name: str, mode: str = "test", **kwargs) -> Run:
        """Returns a `Run` instance created using parameters loaded from MLFlow
        Tracking.

        Args:
            name: Experiment name pattern for filtering.
            mode: Mode name: `'current'`, `'best'`, or `'test'`.

        Examples:
            To load run#4 of the `example` experiment.

                client.load_run_by_name('example', run=4)
        """
        run_id = self.get_run_id(name, **kwargs)
        return self.load_run(run_id, mode)

    def load_instance(self, run_id: str, instance_name: str, mode: str = "test") -> Any:
        """Returns a member of a `Run` created using parameters loaded from MLFlow
        Tracking.

        Args:
            run_id: RunID for a run to be loaded.
            instance_name: Instance name.
            mode: Mode name: `'current'`, `'best'`, or `'test'`.
        """
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
                a `(index, output, target)` and return a tuple with the same signature.
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
            name: Experiment name pattern for filtering.

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
    """Creates an Ivory Client instance.

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
        working directory by MLFlow Tracking.
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
