import datetime
import inspect
from typing import Any, Dict

import optuna
import yaml

import ivory
from ivory import utils
from ivory.callbacks.base import Callback
from ivory.core import instance
from ivory.core.run import Run


class Experiment:
    def __init__(self, run_class: str, shared=None, study=None):
        self.run_class = run_class
        self.run_cls = instance.get_attr(run_class)
        if shared is None:
            shared = []
        self.shared = shared
        self.name = "ready"
        self.experiment_id = ""
        self.num_runs = 0
        self._study = study
        self.study = None

    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"{class_name}(id='{self.experiment_id}', name='{self.name}', "
        s += f"run_class={self.run_class}, num_runs={self.num_runs}, "
        s += f"shared={self.shared})"
        return s

    def params(self, update: Dict[str, Any] = None) -> Dict[str, Any]:
        """Returns a newly created params dictionary.

        Parameters dict is always created from yaml string when this method is called.

        Args:
            update (dict): Update dictionary to overwrite the default settings.

        Returns:
            dict: parameter dictionary for a run
        """
        params = utils.to_float(yaml.safe_load(self.params_yaml))
        if update is None:
            return params
        else:
            utils.update_dict(params, utils.dot_to_list(update))
            return params

    def set_fields(self, params_path, params_yaml):
        """Sets the instance fields that were not initialized in `__init__`.

        Args:
            params_path (str): the yaml parameters file path for this `Experiment`
            params_yaml (str): the yaml body text for this `Experiment`
        """
        self.params_path = params_path
        self.params_yaml = params_yaml
        resolved = instance.resolve_params(self.params(), self.shared)
        self.shared_keys, self.shared = resolved

    def start(self, name=None):
        """Starts this experiment object.

        This method instantiates default objects that will be shared among runs and
        invokes `on_experiment_start` class method of registerd `Callback`s
        """
        params = self.params()
        default = {key: params[key] for key in self.shared_keys}
        self.default = instance.instantiate(default)
        self.default.update(experiment=self)
        self.name = name or self.get_experiment_name()
        for cls in instance.get_classes(params):
            if issubclass(cls, Callback):
                cls.on_experiment_start(self)
        ivory.active_experiment = self

    def create_run(self, update: Dict[str, Any] = None, callbacks=None) -> Run:
        """Creates a run with an optional update parameters.

        Args:
            update (dict): `update` can be used for hyper parameter tuning
            callbacks (list): dynamically created callbacks (ex. optuna pruning)

        Returns:
            Run: a run object
        """
        if self.name == "ready":
            self.start()
        self.num_runs += 1
        name = self.get_run_name()
        params = self.params(update)
        return self.run_cls(
            name=name, params=params, default=self.default, callbacks=callbacks
        )

    def create_study(self):
        """Returns a Optuna Study object."""
        if self._study is None:
            raise ValueError("'study' not found in params file.")
        if self.name == "ready":
            self.start()
        parameters = inspect.signature(optuna.create_study).parameters
        kwargs = {}
        for key in self._study:
            if key in parameters:
                kwargs[key] = self._study[key]
        self.study = optuna.create_study(study_name=self.name, **kwargs)
        self.study.set_user_attr("experiment_id", self.experiment_id)
        return self.study

    def optimize(self):
        """Optimize a Study object."""
        if self.study is None:
            self.create_study()
        objective = create_objective(self)
        parameters = inspect.signature(self.study.optimize).parameters
        kwargs = {}
        for key in self._study:
            if key in parameters:
                kwargs[key] = self._study[key]
        return objective, kwargs

    def get_experiment_name(self):
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def get_run_name(self):
        return f"#{self.num_runs}"


def create_experiment(params_path: str) -> Experiment:
    """Creates an `Experiment` instance from a yaml parameters file.

    Args:
        yaml_params_file (str): yaml parameters file path

    Returns:
        Experiment: an experiment object
    """
    with open(params_path) as file:
        params_yaml = file.read()
    params = utils.to_float(yaml.safe_load(params_yaml))
    experiment = instance.instantiate(params["experiment"])
    experiment.set_fields(params_path, params_yaml)
    return experiment


def create_objective(experiment: Experiment):
    if "objective" not in experiment._study:
        raise ValueError("'objective' not found in 'study' of params file.")
    objective = experiment._study["objective"]
    params = experiment.params()
    monitor = instance.instantiate(params["monitor"]).monitor
    has_pruner = "pruner" in experiment._study

    def _objective(trial):
        objective(trial)
        callbacks = None
        if has_pruner:
            callbacks = [ivory.callbacks.Pruning(trial, monitor)]
        run = experiment.create_run(trial.params, callbacks=callbacks)
        run.name = f"#{trial.number}"
        run.start()
        if "tracking" in run:
            trial.set_user_attr("run_id", run.tracking.run_id)
        return run.monitor.best_score

    return _objective
