import copy
import functools
import itertools

from tqdm import tqdm

from ivory import utils
from ivory.core.base import Base
from ivory.core.instance import create_base_instance, create_instance
from ivory.core.parser import Parser


class Experiment(Base):
    def set_client(self, client):
        if client.tracker:
            self.set_tracker(client.tracker)
        if client.tuner:
            self.set_tuner(client.tuner)

    def set_tracker(self, tracker):
        self["tracker"] = tracker
        if not self.name:
            self.name = "Default"
            self.params["experiment"]["name"] = self.name
        if not self.id:
            self.id = tracker.create_experiment(self.name)
            self.params["experiment"]["id"] = self.id

    def set_tuner(self, tuner):
        self["tuner"] = tuner

    def create_params(self, params=None, **kwargs):
        if not params:
            params = copy.deepcopy(self.params)
            if kwargs:
                update = utils.create_update(params, **kwargs)
                utils.update_dict(params, update)
        if "id" not in params["experiment"] or params["experiment"]["id"] == self.id:
            return params
        raise ValueError("Experiment ids don't match.")

    def create_run(self, params=None, **kwargs):
        params = self.create_params(params, **kwargs)
        run = create_base_instance(params, "run")
        run.set_experiment(self)
        return run

    def create_instance(self, name, params=None, **kwargs):
        params = self.create_params(params, **kwargs)
        if "." not in name:
            name = f"run.{name}"
        return create_instance(params, name)

    def start(self, args=None, repeat=1, message: str = "", **kwargs):
        it = product(args, self.params, repeat=repeat, **kwargs)
        for update, _, mode, number, args, _, tags in it:
            run = self._create_run(update, mode, number, args, tags, message)
            yield run

    def optimize(self, name, args=None, message: str = "", **kwargs):
        params = self.params
        it = product(args, params, repeat=1, desc="Study", **kwargs)
        mode = self.create_instance("run.monitor").mode
        tuner = self.tuner
        optimize = self.objective.optimize
        for update, options, _, _, args, values, tags in it:
            names = sorted([f"{arg}={value}" for arg, value in zip(args, values)])
            extra_name = ",".join(names)
            study_name = ".".join([self.name, name, extra_name])
            tags.update(suggest=name)
            create_study = functools.partial(tuner.create_study, study_name, mode)
            create_run = functools.partial(self._create_run, tags=tags, message=message)
            study = optimize(name, update, params, options, create_run, create_study)
        if self.id:
            study.set_user_attr("experiment_id", self.id)
        yield study

    def _create_run(self, update, mode, number, args, tags, message):
        params = self.create_params()
        utils.update_dict(params, update)
        run_name = "single" if mode == "single" else f"{mode}#{number}"
        params["run"]["name"] = run_name
        run = self.create_run(params)
        if run.tracking:
            args = {arg: utils.get_value(params, arg) for arg in args}
            run.tracking.log_params(run.id, args)
            tags = tags.copy()
            tags["mode"] = mode
            if message:
                tags["message"] = message
            run.tracking.set_tags(run.id, tags)
        return run

    def search_runs(self, **query):
        for run_id in self.tracker.list_run_ids(self.id):
            params = self.load_params(run_id)
            if utils.match(params, **query):
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


def product(args, params, repeat=1, desc="Run  ", **kwargs):
    parser = Parser(params, args, **kwargs)
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
