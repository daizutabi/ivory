import copy
from typing import Iterator

from ivory import utils
from ivory.core.base import Base
from ivory.core.default import DEFAULT_CLASS
from ivory.core.instance import create_base_instance, create_instance


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

    def get_run_name(self, class_name: str, run_number: int = 0):
        if run_number == 0:
            if self.tracker:
                for run_id in self.search_runs(run_view_type=3):
                    name = self.tracker.get_run_name(run_id)
                    if name.startswith(class_name):
                        run_number = max(run_number, int(name.split("#")[1]))
            run_number += 1
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
            params[name] = {"class": DEFAULT_CLASS["core"][name]}
        params[name]["name"] = self.get_run_name(class_name, run_number)
        run = create_base_instance(params, name)
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
        return create_instance(params, name)

    def search_runs(self, run_view_type=1, **query) -> Iterator[str]:
        for run_id in self.tracker.list_run_ids(self.id, run_view_type):
            if query:
                params = self.load_params(run_id)
                if utils.match(params, **query):
                    yield run_id
            else:
                yield run_id

    def load_params(self, run_id):
        return self.tracker.load_params(run_id)

    def load_run(self, run_id, mode="test"):
        return self.tracker.load_run(run_id, mode, self.create_run)

    def load_instance(self, run_id, name, mode="test"):
        return self.tracker.load_instance(
            run_id, name, mode, self.create_run, self.create_instance
        )

    def update_params(self, **default):
        self.tracker.update_params(self.id, **default)
