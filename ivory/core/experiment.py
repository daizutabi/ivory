import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ivory import utils
from ivory.core import instance
from ivory.core.tracker import Tracker
from ivory.core.tuner import Tuner


@dataclass
class Experiment:
    run_class: str
    shared: List[str] = field(default_factory=list)
    tracker: Optional[Tracker] = None
    tuner: Optional[Tuner] = None

    def __post_init__(self):
        self.run_cls = instance.get_attr(self.run_class)
        self.experiment_id = ""
        self.name = "ready"
        self.num_runs = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"{class_name}(id='{self.experiment_id}', name='{self.name}', "
        s += f"run_class='{self.run_class}', num_runs={self.num_runs}, "
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

    def start(self):
        """Starts the experiment.

        This method instantiates default objects that will be shared among runs.
        """
        params = self.params()
        shared_params = {key: params[key] for key in self.shared_keys}
        self.shared_objects = instance.instantiate(shared_params)
        self.shared_objects.update(experiment=self)
        self.name = self.get_experiment_name()
        if self.tracker:
            self.experiment_id = self.tracker.create_experiment(self.name)
        if self.tuner:
            study = self.tuner.create_study(self.name, params["monitor"])
            if self.experiment_id:
                study.set_user_attr("experiment_id", self.experiment_id)

    def create_run(self, update: Dict[str, Any] = None, callbacks=None, name=None):
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
        if name is None:
            name = self.get_run_name()
        params = self.params(update)
        if callbacks is None:
            callbacks = []
        if self.tracker:
            callbacks += [self.tracker.create_callback(self.experiment_id)]
        return self.run_cls(name, params, self.shared_objects, callbacks)

    def optimize(self):
        self.tuner.optimize(self.create_run)

    def get_experiment_name(self):
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def get_run_name(self):
        return f"#{self.num_runs}"


def create_experiment(params_path: str, update: Dict[str, Any] = None):
    """Creates an `Experiment` instance from a yaml parameters file.

    Args:
        yaml_params_file (str): yaml parameters file path

    Returns:
        Experiment: an experiment object
    """
    with open(params_path) as file:
        params_yaml = file.read()
    params = utils.to_float(yaml.safe_load(params_yaml))
    if update:
        utils.update_dict(params, utils.dot_to_list(update))
        params_yaml = yaml.dump(params, sort_keys=False)
    experiment = instance.instantiate(params["experiment"])
    experiment.set_fields(params_path, params_yaml)
    return experiment
