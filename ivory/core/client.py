from typing import Dict

import ivory.core.ui
from ivory import utils
from ivory.core.base import Base
from ivory.core.experiment import Experiment
from ivory.core.instance import create_base_instance


class Client(Base):
    def __init__(self, *args, **kwargs):
        super(Client, self).__init__(*args, **kwargs)
        self.experiments: Dict[str, Experiment] = {}
        self.run_id_experiment: Dict[str, Experiment] = {}

    def create_params(self, path: str):
        return utils.load_params(path, self.source_name)[0]

    def create_experiment(self, path: str) -> Experiment:
        params, source_name = utils.load_params(path, self.source_name)
        return self.create_experiment_from_params(params, source_name)

    def create_experiment_from_params(self, params, source_name=""):
        experiment = create_base_instance(params, "experiment", source_name)
        experiment.set_client(self)
        return experiment

    def get_experiments(self, path="", exists_only=True):
        for params, source_name in utils.params_iter(self.source_name, path):
            if source_name in self.experiments:
                yield self.experiments[source_name]
                continue
            name = params["experiment"]["name"]
            if exists_only and not self.tracker.get_experiment_id(name):
                continue
            experiment = self.create_experiment_from_params(params, source_name)
            self.experiments[source_name] = experiment
            yield experiment

    def search_runs(self, path="", **query):
        for experiment in self.get_experiments(path):
            for run_id in experiment.search_runs(**query):
                self.run_id_experiment[run_id] = experiment
                yield run_id

    def get_experiment_from_run_id(self, run_id):
        if run_id not in self.run_id_experiment:
            msg = "Unknown a run_id. You have to get a run_id from client.search_runs."
            raise ValueError(msg)
        return self.run_id_experiment[run_id]

    def load_params(self, run_id):
        return self.tracker.load_params(run_id)

    def load_run(self, run_id, mode="test"):
        experiment = self.get_experiment_from_run_id(run_id)
        return self.tracker.load_run(run_id, mode, experiment.create_run)

    def load_instance(self, run_id, name, mode="test"):
        experiment = self.get_experiment_from_run_id(run_id)
        return self.tracker.load_instance(
            run_id, name, mode, experiment.create_run, experiment.create_instance
        )

    def ui(self):
        ivory.core.ui.run(self.tracker.tracking_uri)


def create_client(path="client", directory=".", tracker=True) -> Client:
    source_name = utils.normpath(path, directory)
    params, _ = utils.load_params(source_name)
    if not tracker and "tracker" in params["client"]:
        params["client"].pop("tracker")
    with utils.chdir(source_name):
        client = create_base_instance(params, "client", source_name)
    return client
