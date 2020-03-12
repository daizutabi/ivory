import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

import ivory
from ivory.callbacks.base import Callback
from ivory.core.instance import get_attr, get_classes, instantiate, resolve_params
from ivory.core.run import Run
from ivory.utils import dot_to_list, to_float, update_dict


@dataclass
class Experiment:
    name: str = "ready"
    run_class: str = "ivory.core.Run"
    shared: List[str] = field(default_factory=list)
    num_runs: int = 0

    def __post_init__(self):
        self.run_cls = get_attr(self.run_class)
        self.params_path = None
        self.yaml = None
        self.default = None
        self.shared_keys = None

    def params(self, update: Dict[str, Any] = None) -> Dict[str, Any]:
        """Returns a newly created params dictionary.

        Parameters dict is always created from yaml string when this method is called.

        Args:
            update (dict): Update dictionary to overwrite the default settings.

        Returns:
            dict: parameter dictionary for a run
        """
        params = to_float(yaml.safe_load(self.yaml))
        if update is None:
            return params
        else:
            update_dict(params, dot_to_list(update))
            return params

    def set_fields(self, params_path, yaml):
        """Sets the instance fields that were not initialized in `__init__`.

        Args:
            params_path (str): the yaml parameters file path for this `Experiment`
            yaml (str): the yaml body text for this `Experiment`
        """
        self.params_path, self.yaml = params_path, yaml
        self.shared_keys, self.shared = resolve_params(self.params(), self.shared)

    def start(self):
        """Starts this experiment object.

        This method instantiates default objects that will be shared among runs and
        invokes `on_experiment_start` class method of registerd `Callback`s
        """
        params = self.params()
        self.default = instantiate({key: params[key] for key in self.shared_keys})
        self.default.update(experiment=self)
        self.name = self.get_experiment_name()
        for cls in get_classes(params):
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
        if callbacks is None:
            callbacks = []
        return self.run_cls(
            name=name, params=params, default=self.default, callbacks=callbacks
        )

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
        yml = file.read()
    params = to_float(yaml.safe_load(yml))
    experiment = instantiate(params["experiment"])
    experiment.set_fields(params_path, yml)
    return experiment
