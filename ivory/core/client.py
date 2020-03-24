import copy
import itertools
import os
import subprocess
import tempfile
from typing import List

from ivory import utils
from ivory.core.instance import create_base_instance, create_instance


class Client:
    def __init__(self, params, source_name=""):
        if isinstance(params, str):
            source_name = os.path.abspath(params)
            params = utils.load_params(params)
        self.params = params
        self.source_name = source_name
        self.tracker = None
        self.create_experiment()

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({self.experiment})"

    def create_experiment(self, params=None):
        params = params or self.params
        experiment = create_base_instance(params, "experiment", self.source_name)
        if "environment" in self.params:
            env = create_base_instance(params, "environment", self.source_name)
            experiment.set_environment(env)
        self.experiment = experiment
        if experiment.tracker:
            self.tracker = experiment.tracker
        return experiment

    def create_run(self, params=None):
        params = params or self.params
        run = create_base_instance(params, "run", self.source_name)
        run.set_experiment(self.experiment)
        return run

    def create_instance(self, name):
        return create_instance(self.params, name)

    def product(self, args: List[str], message: str = ""):
        args_dict = utils.parse_args(self.params["run"], args)
        if len(args) == 0 or all([len(x) == 1 for x in args_dict.values()]):
            mode = "single"
        elif len(args) == 1:
            mode = "scan"
        else:
            mode = "product"
        number = 1
        for value in itertools.product(*args_dict.values()):
            update = {key: value for key, value in zip(args_dict.keys(), value)}
            self._update_run_start(update, mode, number, args, args_dict, message)
            number += 1

    def chain(self, args: List[str], message: str = ""):
        args_dict = utils.parse_args(self.params["run"], args)
        mode = "chain"
        number = 1
        for name in args_dict:
            for value in args_dict[name]:
                update = {name: value}
                self._update_run_start(update, mode, number, args, args_dict, message)
                number += 1

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])

    def search_runs(self, params=None, tags=None, **kwargs):
        if kwargs:
            params = kwargs
        return self.tracker.search_runs(self.experiment.id, params, tags)

    def list(self, args: List[str], message: str = ""):
        filter_params = {}
        filter_tags = {}
        for arg in args:
            if "=" not in arg:
                filter_tags["mode"] = arg
            else:
                key, value = arg.split("=")
                filter_params[key] = value
        if message:
            filter_tags["message"] = message
        return self.search_runs(filter_params, filter_tags)

    def load_run(self, run_id, epoch="best"):
        client = self.tracker.client
        if epoch == "best":
            for artifact in client.list_artifacts(run_id):
                if artifact.is_dir and artifact.path == "best":
                    break
            else:
                epoch = "current"
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = client.download_artifacts(run_id, "params.yaml", tmpdir)
            state_dict_path = client.download_artifacts(run_id, epoch, tmpdir)
            params = utils.load_params(params_path)
            run = self.create_run(params)
            state_dict = run.load(state_dict_path)
            run.load_state_dict(state_dict)
        return run

    def _update_run_start(self, update, mode, number, args, args_dict, message):
        params = copy.deepcopy(self.params["run"])
        utils.update_dict(params, update)
        params["name"] = f"{mode}#{number}"
        run = self.create_run(params)
        if run.tracking:
            run.tracking.param_names = list(args_dict.keys())
            tags = {"message": message} if message else {}
            tags["mode"] = mode
            for arg in args:
                key, value = arg.split("=")
                tags[key] = value
            run.tracking.set_tags(run.id, tags)
        run.start()
