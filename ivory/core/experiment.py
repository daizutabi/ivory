from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

import ivory
from ivory.callbacks.base import Callback
from ivory.core.instance import get_attr, get_classes, instantiate
from ivory.core.run import Run
from ivory.utils import dot_to_list, format_name_by_dict, to_float, update_dict


@dataclass
class Experiment:
    name: str
    run_class: str
    run_name: str
    shared: List[str] = field(default_factory=list)
    yaml: str = field(default="", repr=False)
    default: Dict[str, Any] = field(default_factory=dict, repr=False)

    def set_yaml(self, yaml):
        self.yaml = yaml
        self.name = format_name_by_dict(self.name, self.params())
        if self.shared:
            self.set_default(self.shared)

    def start(self):
        for cls in get_classes(self.params()):
            if issubclass(cls, Callback):
                cls.on_experiment_start(self)
        ivory.active_experiment = self

    def params(self, update: Dict[str, Any] = None) -> Dict[str, Any]:
        """Return a newly created params dictionary for each run.

        Config is always created from yaml string when this method is called.
        """
        params = to_float(yaml.safe_load(self.yaml))
        if update is None:
            return params
        else:
            update_dict(params, dot_to_list(update))
            return params

    def set_default(self, names: List[str]):
        """Set default objects which are shared for every run."""
        params = self.params()
        self.default = instantiate(params, names=names)

    def create_run(self, update: Dict[str, Any] = None, callbacks=None) -> Run:
        """Create a run for an optional update params.

        Args:
            update: dict can be used for hyper parameter tuning.
            callbacks: dynamically created callbacks (for example optuna pruning)
        """
        params = self.params(update)
        cls = get_attr(self.run_class)
        if "experiment" not in self.default:
            self.default.update(experiment=self)
        run = cls(params, default=self.default, callbacks=callbacks)
        run_name = params["experiment"]["run_name"]
        run.name = format_name_by_dict(run_name, params)
        return run


def create_experiment(yaml_params_file: str) -> Experiment:
    """Create an Objective instance from a yaml params file.

    Parameters
    ----------
    yaml_params_file : str
        Yaml params file path.

    Returns
    -------
    Objective instance.
    """
    with open(yaml_params_file) as file:
        yml = file.read()
    params = to_float(yaml.safe_load(yml))
    experiment = instantiate(params["experiment"])
    experiment.set_yaml(yml)
    return experiment
