import copy
import datetime
from typing import Any, Dict

import yaml

from ivory import utils
from ivory.core import instance
from ivory.core.base import Base


class Experiment(Base):
    __slots__ = ["num_runs"]

    def __init__(self, params, **objects):
        super().__init__("ready", params, **objects)
        self.num_runs = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        run_class = self.params["run"]["class"]
        s = f"{class_name}(id='{self.id}', name='{self.name}', "
        s += f"run_class='{run_class}', num_runs={self.num_runs})"
        return s

    def start(self):
        """Starts the experiment.

        This method instantiates global objects that will be shared among runs.
        """
        self.name = self.get_experiment_name()
        if self.tracker:
            self.id = self.tracker.create_experiment(self.name)
        if self.tuner:
            study = self.tuner.create_study(self.name, self.params["run"]["monitor"])
            if self.id:
                study.set_user_attr("experiment_id", self.id)

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

        params = copy.deepcopy(self.params)
        if update:
            utils.update_dict(params["run"], update)
        if callbacks is None:
            callbacks = {}
        if self.tracker:
            callbacks["tracking"] = self.tracker.create_tracking(self.id)

        kwargs = dict(name=name, params=params, **callbacks)
        return instance.instantiate(params["run"], kwargs=kwargs)

    def optimize(self):
        self.tuner.optimize(self.create_run)

    def get_experiment_name(self):
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def get_run_name(self):
        return f"#{self.num_runs}"


def create_experiment(params, update: Dict[str, Any] = None):
    """Creates an `Experiment` instance.

    Args:
        update (dict): update params

    Returns:
        Experiment: an experiment object
    """
    if isinstance(params, str):
        path = params
        with open(path, "r") as file:
            params_yaml = file.read()
        params = utils.to_float(yaml.safe_load(params_yaml))
    if update:
        utils.update_dict(params, update)
    return instance.instantiate(params["experiment"], kwargs={"params": params})
