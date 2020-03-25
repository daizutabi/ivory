import copy
import functools
import itertools
import os
import subprocess
import tempfile
from typing import List

from ivory import utils
from ivory.core.base import Base
from ivory.core.instance import create_base_instance, create_instance
from ivory.core.parser import Parser


def create_client(params, source_name=""):
    if isinstance(params, str):
        source_name = os.path.abspath(params)
        params = utils.load_params(params)
    return create_base_instance(params, "client", source_name)


class Client(Base):
    __slots__ = []  # type:ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_experiment()
        self.params.pop("client")

    def create_experiment(self, params=None):
        params = params or self.params
        experiment = create_base_instance(params, "experiment", self.source_name)
        experiment.set_client(self)
        self.objects["experiment"] = experiment
        return experiment

    def create_run(self, params=None):
        params = params or self.params
        run = create_base_instance(params, "run", self.source_name)
        run.set_experiment(self.experiment)
        return run

    def create_instance(self, name):
        return create_instance(self.params, name)

    def product(self, args, repeat=1, message: str = ""):
        parser = Parser().parse(args, self.params["run"])
        if repeat != 1 and parser.mode == "single":
            parser.mode = "repeat"
        args = parser.args.keys()
        tags = parser.args
        number = 1
        for _ in range(repeat):
            for value in itertools.product(*parser.values):
                update = dict(zip(parser.names, value))
                run = self._create_run(update, parser.mode, number, args, tags, message)
                yield run
                number += 1

    def chain(self, args, repeat=1, message: str = ""):
        parser = Parser().parse(args, self.params["run"])
        mode = "chain"
        args = parser.args.keys()
        tags = parser.args
        number = 1
        for _ in range(repeat):
            for k, name in enumerate(parser.names):
                for value in parser.values[k]:
                    update = {name: value}
                    run = self._create_run(update, mode, number, args, tags, message)
                    yield run
                    number += 1

    def optimize(self, name, options, message: str = ""):
        if name is None:
            name = list(self.experiment.objective.suggest.keys())[0]

        if "delete" in options:
            study_name = ".".join([self.experiment.name, options["delete"]])
            self.tuner.delete_study(study_name)
            return

        create_run = functools.partial(self._create_run, message=message)
        create_objective = self.experiment.objective.create_objective
        has_pruner = self.pruner is not None
        objective = create_objective(name, self.params, create_run, has_pruner)
        mode = self.create_instance("run.monitor").mode
        study_name = ".".join([self.experiment.name, name])
        study = self.tuner.create_study(study_name, mode, self.experiment.id)
        study.optimize(objective, **options)

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        try:
            subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri])
        except KeyboardInterrupt:
            pass

    def search_runs(self, mode, params, message: str = ""):
        tags = {}
        if mode:
            tags["mode"] = mode
        if message:
            tags["message"] = message
        for value in itertools.product(*params.values()):
            params_ = dict(zip(params.keys(), value))
            yield from self.tracker.search_runs(self.experiment.id, params_, tags)

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

    def _create_run(self, update, mode, number, args, tags, message):
        params = copy.deepcopy(self.params)
        utils.update_dict(params["run"], update)
        run_name = "single" if mode == "single" else f"{mode}#{number}"
        params["run"]["name"] = run_name
        run = self.create_run(params)
        if run.tracking:
            args = {arg: utils.get_value(params["run"], arg) for arg in args}
            run.tracking.log_params(run.id, args)
            tags = tags.copy()
            tags["mode"] = mode
            if message:
                tags["message"] = message
            run.tracking.set_tags(run.id, tags)
        return run
