import os
import re
import subprocess
from typing import Any, Dict, Iterator, Tuple

import ivory.utils.data
from ivory import utils
from ivory.core import default, instance
from ivory.core.base import Base
from ivory.core.experiment import Experiment
from ivory.core.run import Run
from ivory.utils.tqdm import tqdm


class Client(Base):
    def create_experiment(self, path: str, name: str = "") -> Experiment:
        params, source_name = utils.load_params(path, self.source_name)
        if "experiment" not in params:
            params.update(default.get("experiment"))
        if "name" not in params["experiment"]:
            if name:
                path = ".".join([path, name])
            params["experiment"]["name"] = path
        experiment = instance.create_base_instance(params, "experiment", source_name)
        if self.tracker:
            experiment.set_tracker(self.tracker)
        return experiment

    def search_run_ids(
        self,
        name: str = "",
        parent_run_id: str = "",
        parent_only: bool = False,
        **query
    ) -> Iterator[str]:
        for experiment in self.tracker.list_experiments():
            if name and not re.match(name, experiment.name):
                continue
            yield from self.tracker.search_run_ids(
                experiment.experiment_id, parent_run_id, parent_only, **query
            )

    def load_params(self, run_id: str) -> Dict[str, Any]:
        return self.tracker.load_params(run_id)

    def load_run(self, run_id: str, mode: str = "test") -> Run:
        run = self.tracker.load_run(run_id, mode)
        run.set_tracker(self.tracker)
        return run

    def load_instance(self, run_id: str, name: str, mode: str = "test") -> Any:
        return self.tracker.load_instance(run_id, name, mode)

    def load_results(self, run_ids: str, verbose: bool = True) -> Tuple:
        if verbose:
            run_ids = tqdm(list(run_ids))
        it = (self.load_instance(run_id, "results", "test") for run_id in run_ids)
        return ivory.utils.data.concat_results(it)

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        try:
            subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])
        except KeyboardInterrupt:
            pass


def create_client(path="client", directory=".", tracker=True) -> Client:
    source_name = utils.normpath(path, directory)
    if os.path.exists(source_name):
        params, _ = utils.load_params(source_name)
    else:
        params = default.get("client")
    if not tracker and "tracker" in params["client"]:
        params["client"].pop("tracker")
    with utils.chdir(source_name):
        client = instance.create_base_instance(params, "client", source_name)
    return client
