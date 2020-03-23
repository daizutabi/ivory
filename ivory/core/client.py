import copy
import itertools
import os
import subprocess
import tempfile
from typing import List

from ivory import utils
from ivory.core.instance import create_base_instance


class Client:
    def __init__(self, params, source_name=""):
        if isinstance(params, str):
            source_name = os.path.abspath(params)
            params = utils.load_params(params)
        self.params = params
        self.source_name = source_name
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
        else:
            self.tracker = None
        return experiment

    def create_run(self, params=None):
        params = params or self.params
        run = create_base_instance(params, "run", self.source_name)
        run.set_experiment(self.experiment)
        return run

    def chain(self, args: List[str], message: str = ""):
        params = self.params["run"]
        args_dict = utils.parse_args(params, args)
        number = 1
        for name in args_dict:
            for value in args_dict[name]:
                params_chain = copy.deepcopy(params)
                utils.update_dict(params_chain, {name: value})
                params_chain["name"] = f"chain#{number}"
                run = self.create_run(params_chain)
                set_param_and_tags(run, "chain", args, args_dict.keys(), message)
                run.start()
                number += 1

    def product(self, args: List[str], message: str = ""):
        params = self.params["run"]
        args_dict = utils.parse_args(params, args)
        if len(args) == 0 or all([len(x) == 1 for x in args_dict.values()]):
            run_name = "single"
        elif len(args) == 1:
            run_name = "scan"
        else:
            run_name = "product"
        number = 1
        for value in itertools.product(*args_dict.values()):
            params_prod = copy.deepcopy(params)
            update = {key: value for key, value in zip(args_dict.keys(), value)}
            utils.update_dict(params_prod, update)
            params_prod["name"] = f"{run_name}#{number}"
            run = self.create_run(params_prod)
            set_param_and_tags(run, run_name, args, args_dict.keys(), message)
            run.start()
            number += 1

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])

    def search_runs(self, filter_params, filter_tags):
        return self.tracker.search_runs(self.experiment.id, filter_params, filter_tags)

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


def set_param_and_tags(run, mode, args, params, message):
    if not run.tracking:
        return
    run.tracking.param_names = list(params)
    tags = {"message": message} if message else {}
    tags["mode"] = mode
    for arg in args:
        key, value = arg.split("=")
        tags[key] = value
    run.tracking.set_tags(run.id, tags)
