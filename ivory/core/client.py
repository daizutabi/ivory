import os
from typing import Dict

import ivory.core.ui
import ivory.utils.data
from ivory import utils
from ivory.core import instance
from ivory.core.base import Base
from ivory.core.default import DEFAULT_CLASS
from ivory.core.experiment import Experiment
from ivory.utils.tqdm import tqdm


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
        experiment = instance.create_base_instance(params, "experiment", source_name)
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

    def load_results(self, run_ids, verbose=True):
        if verbose:
            run_ids = tqdm(list(run_ids))
        it = (self.load_instance(run_id, "results", "test") for run_id in run_ids)
        return ivory.utils.data.concat_results(it)

    def ui(self):
        ivory.core.ui.run(self.tracker.tracking_uri)


def create_client(path="client", directory=".", tracker=True) -> Client:
    source_name = utils.normpath(path, directory)
    if os.path.exists(source_name):
        params, _ = utils.load_params(source_name)
        if not tracker and "tracker" in params["client"]:
            params["client"].pop("tracker")
    else:
        params = {"client": {"class": DEFAULT_CLASS["core"]["client"]}}
        if tracker:
            tracker = {"tracker": {"class": DEFAULT_CLASS["core"]["tracker"]}}
            params["client"].update(tracker)
    with utils.chdir(source_name):
        client = instance.create_base_instance(params, "client", source_name)
    return client
