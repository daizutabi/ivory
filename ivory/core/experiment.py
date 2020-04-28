import copy

from ivory import utils
from ivory.core import instance
from ivory.core.base import Base


class Experiment(Base):
    def set_client(self, client):
        if client.tracker:
            self.set_tracker(client.tracker)

    def set_tracker(self, tracker):
        self["tracker"] = tracker
        if not self.name:
            self.name = "Default"
            self.params["experiment"]["name"] = self.name
        if not self.id:
            self.id = tracker.create_experiment(self.name)
            self.params["experiment"]["id"] = self.id

    def create_run_name(self, class_name: str, run_number: int = 0):
        run_number == 0
        if self.tracker:
            run_number = self.tracker.get_run_number(self.id, class_name)
        return f"{class_name}#{run_number:03d}"

    def create_params(self, params=None, args=None, **kwargs):
        if params is None:
            params = copy.deepcopy(self.params)
        update, args = utils.create_update(params["run"], args, **kwargs)
        utils.update_dict(params["run"], update)
        if "id" not in params["experiment"] or params["experiment"]["id"] == self.id:
            return params, args
        raise ValueError("Experiment ids don't match.")

    def create_run(
        self, params=None, class_name="Run", run_number=0, args=None, **kwargs
    ):
        params, args = self.create_params(params, args, **kwargs)
        name = class_name.lower()
        if name not in params:
            params[name] = {}
        params[name]["name"] = self.create_run_name(class_name, run_number)
        run = instance.create_base_instance(params, name)
        run.set_experiment(self)
        if run.tracking:
            args = {arg: utils.get_value(run.params["run"], arg) for arg in args}
            run.tracking.log_params(run.id, args)
        return run

    def create_task(self):
        return self.create_run(class_name="Task")

    def create_study(self, run_number: int = 0):
        return self.create_run(class_name="Study", run_number=run_number)

    def create_instance(self, name: str, params=None, args=None, **kwargs):
        params, _ = self.create_params(params, args, **kwargs)
        if "." not in name:
            name = f"run.{name}"
        return instance.create_instance(params, name)

    def update_params(self, **default):
        self.tracker.update_params(self.id, **default)
