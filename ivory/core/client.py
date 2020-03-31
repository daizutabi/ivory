import copy
import functools
import itertools
import os
from collections.abc import Iterable

from tqdm import tqdm

import ivory.core.ui
from ivory import utils
from ivory.core.base import Base
from ivory.core.instance import create_base_instance, create_instance
from ivory.core.parser import Parser


def create_client(path="client", directory="."):
    source_name = utils.normpath(path, directory)
    params = utils.load_params(source_name)
    with utils.chdir(source_name):
        return create_base_instance(params, "client", source_name)


class Client(Base):
    def create_experiment(self, path: str):
        directory = os.path.dirname(self.source_name)
        source_name = utils.normpath(path, directory)
        params = utils.load_params(source_name)
        experiment = create_base_instance(params, "experiment", source_name)
        experiment.set_client(self)
        self.set_experiment(experiment)
        return experiment

    def set_experiment(self, experiment):
        self["experiment"] = experiment
        self.params = experiment.params

    def create_params(self, params=None):
        if not params:
            return copy.deepcopy(self.params)
        if params["experiment"]["id"] != self.experiment.id:
            raise ValueError("Experiment ids don't match.")
        return params

    def create_run(self, params=None):
        params = self.create_params(params)
        run = create_base_instance(params, "run")
        run.set_experiment(self.experiment)
        return run

    def create_instance(self, name, params=None):
        params = self.create_params(params)
        if "." not in name:
            name = f"run.{name}"
        return create_instance(params, name)

    def run(self, args=None, repeat=1, message: str = "", **kwargs):
        it = product(args, self.params["run"], repeat, **kwargs)
        for update, _, mode, number, args, _, tags in it:
            run = self._create_run(update, mode, number, args, tags, message)
            yield run

    def optimize(self, name, args=None, message: str = "", **kwargs):
        it = product(args, self.params["run"], repeat=1, desc="Study", **kwargs)
        mode = self.create_instance("run.monitor").mode
        tuner = self.tuner
        optimize = self.experiment.objective.optimize
        params = self.params
        for update, options, _, _, args, values, tags in it:
            names = sorted([f"{arg}={value}" for arg, value in zip(args, values)])
            extra_name = ",".join(names)
            study_name = ".".join([self.experiment.name, name, extra_name])
            tags.update(suggest=name)
            create_study = functools.partial(tuner.create_study, study_name, mode)
            create_run = functools.partial(self._create_run, tags=tags, message=message)
            study = optimize(name, update, params, options, create_run, create_study)
        if self.experiment.id:
            study.set_user_attr("experiment_id", self.experiment.id)
        yield study

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        ivory.core.ui.run(tracking_uri)

    def list_run_ids(self):
        return self.tracker.list_run_ids(self.experiment.id)

    def search_runs(self, params=None, tags=None, return_id=True, **kwargs):
        id = self.experiment.id
        if params is None:
            params = {}
        else:
            params = params.copy()
        for key, value in kwargs.items():
            if isinstance(value, str) or not isinstance(value, Iterable):
                value = [value]
            params[key] = value
        if not params:
            yield from self.tracker.search_runs(id, None, tags, return_id)
            return
        for value in itertools.product(*params.values()):
            params_ = dict(zip(params.keys(), value))
            yield from self.tracker.search_runs(id, params_, tags, return_id)

    def load_params(self, run_id):
        return self.tracker.load_params(run_id)

    def load_run(self, run_id, mode="test"):
        return self.tracker.load_run(run_id, mode, self.create_run)

    def load_runs(self, run_ids, mode="test"):
        for run_id in run_ids:
            yield self.load_run(run_id, mode)

    def load_instance(self, run_id, name, mode="test"):
        return self.tracker.load_instance(
            run_id, name, mode, self.create_run, self.create_instance
        )

    def load_instances(self, run_ids, name, mode="test"):
        for run_id in run_ids:
            yield self.load_instances(run_ids, name, mode)

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

    def update_params(self, **kwargs):
        self.tracker.update_params(self.experiment.id, **kwargs)


def product(args, params, repeat=1, desc="Run  ", **kwargs):
    parser = Parser().parse(args, params, **kwargs)
    if repeat != 1 and parser.mode == "single":
        parser.mode = "repeat"
    args = parser.fullnames.keys()
    tags = parser.args
    options = parser.options
    it = list(itertools.product(range(repeat), *parser.update.values()))
    if len(it) > 1:
        it = tqdm(it, desc=desc)
    for number, (_, *values) in enumerate(it, 1):
        update = {}
        for fullnames, value in zip(parser.update, values):
            for fullname in fullnames:
                update[fullname] = value
        yield update, options, parser.mode, number, args, values, tags
