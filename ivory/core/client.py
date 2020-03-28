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


def create_client(params, source_name=""):
    if isinstance(params, str):
        source_name = os.path.abspath(params)
        params = utils.load_params(params)
    with utils.chdir(source_name):
        return create_base_instance(params, "client", source_name)


class Client(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "experiment" in self.params:
            self.create_experiment()
        self.params.pop("client")

    def create_experiment(self, params=None):
        params = params or self.params
        experiment = create_base_instance(params, "experiment", self.source_name)
        experiment.set_client(self)
        self["experiment"] = experiment
        return experiment

    def create_run(self, params=None):
        params = params or self.params
        run = create_base_instance(params, "run", self.source_name)
        run.set_experiment(self.experiment)
        return run

    def create_instance(self, name, params=None):
        params = params or self.params
        if "." not in name:
            name = f"run.{name}"
        return create_instance(params, name)

    def product(self, args, repeat=1, message: str = ""):
        parser = Parser().parse(args, self.params["run"])
        if repeat != 1 and parser.mode == "single":
            parser.mode = "repeat"
        args = parser.args.keys()
        tags = parser.args
        it = itertools.product(range(repeat), *parser.values)
        total = repeat
        for value in parser.values:
            total *= len(value)
        number = 1
        for _, *value in tqdm(it, total=total, desc="Run  "):
            update = {}
            for names, v in zip(parser.names, value):
                for name in names:
                    update[name] = v
            run = self._create_run(update, parser.mode, number, args, tags, message)
            yield run
            number += 1

    def chain(self, args, repeat=1, message: str = ""):
        parser = Parser().parse(args, self.params["run"])
        mode = "chain"
        args = parser.args.keys()
        tags = parser.args
        total = repeat * sum(len(value) for value in parser.values)
        number = 1
        it = itertools.product(range(repeat), enumerate(parser.names))
        for _, (k, names) in tqdm(it, total=total, desc="Run  "):
            for value in parser.values[k]:
                update = {name: value for name in names}
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
        objective = create_objective(name, self.params, create_run)
        mode = self.create_instance("run.monitor").mode
        study_name = ".".join([self.experiment.name, name])
        pruner = self.experiment.objective.pruner
        sampler = self.experiment.objective.sampler
        study = self.tuner.create_study(
            study_name, mode, self.experiment.id, sampler=sampler, pruner=pruner
        )
        study.optimize(objective, **options)

    def ui(self):
        tracking_uri = self.tracker.tracking_uri
        ivory.core.ui.run(tracking_uri)

    def search_runs(self, params=None, mode="", message="", return_id=True, **kwargs):
        tags = {}
        if mode:
            tags["mode"] = mode
        if message:
            tags["message"] = message
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

    def load_run(self, run_id, mode="test"):
        return self.tracker.load_run(run_id, mode, self.create_run)

    def load_runs(self, run_ids, mode="test"):
        for run_id in run_ids:
            yield self.load_run(run_id, mode)

    def load_instance(self, run_id, name, mode="test"):
        return self.tracker.load_instance(
            run_id, name, mode, self.create_run, self.create_instance
        )

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
