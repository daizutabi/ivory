import copy
import datetime
from typing import Any, Dict

from ivory import utils
from ivory.core import instance
from ivory.core.base import Base


class Experiment(Base):
    def __repr__(self):
        class_name = self.__class__.__name__
        run_class = self.params["run"]["class"]
        s = f"{class_name}(id='{self.id}', name='{self.name}', "
        s += f"run_class='{run_class}')"
        return s

    def start(self):
        """Starts the experiment.

        This method instantiates global objects that will be shared among runs.
        """
        if not self.name:
            self.name = self.get_experiment_name()
            self.params["experiment"]["name"] = self.name
        if self.tracker:
            self.id = self.tracker.create_experiment(self.name)
            self.params["experiment"]["id"] = self.id

    def create_run(self, update: Dict[str, Any] = None, name=None, objects=None):
        """Creates a run with an optional update parameters.

        Args:
            update (dict): `update` can be used for hyper parameter tuning
            objects (dict): dynamically created objects (ex. optuna pruning)

        Returns:
            Run: a run object
        """
        params = copy.deepcopy(self.params)
        if update:
            utils.update_dict(params["run"], update)
        if name:
            if "name" in params["run"]:
                raise ValueError("Run is already named.")
            else:
                params["run"]["name"] = name
        if objects is None:
            objects = {}
        if self.tracker:
            objects["tracking"] = self.tracker.create_tracking(self.id)
        kwargs = dict(params=params, **objects)
        globals = self.objects.copy()
        return instance.instantiate(params["run"], globals=globals, kwargs=kwargs)

    def optimize(self):
        if self.tuner and self.tuner.study is None:
            study = self.tuner.create_study(self.name, self.params["run"]["monitor"])
            if self.id:
                study.set_user_attr("experiment_id", self.id)
        self.tuner.optimize(self.create_run)

    def get_experiment_name(self):
        return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def create_experiment(params, update: Dict[str, Any] = None):
    """Creates an `Experiment` instance.

    Args:
        update (dict): update params

    Returns:
        Experiment: an experiment object
    """
    if isinstance(params, str):
        params = utils.load_params(params)
    if update:
        utils.update_dict(params, update)
    return instance.instantiate(params["experiment"], kwargs=dict(params=params))
