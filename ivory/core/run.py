import os

import ivory.core.state
from ivory.core import parser
from ivory.core.base import CallbackCaller
from ivory.utils.tqdm import tqdm


class Run(CallbackCaller):
    def set_experiment(self, experiment):
        if experiment.source_name:
            self.source_name = experiment.source_name
        if experiment.tracker:
            self.set_tracker(experiment.tracker, experiment.id)
        self["experiment"] = experiment

    def set_tracker(self, tracker, experiment_id: str = ""):
        if not self.id:
            self.id = tracker.create_run(experiment_id, self.name, self.source_name)
            class_name = self.__class__.__name__.lower()
            self.params[class_name]["id"] = self.id
        self["tracking"] = tracker.create_tracking()

    def init(self, mode: str = "train"):
        self.create_callbacks()
        self.mode = mode
        self.on_init_start()
        self.on_init_end()

    def start(self, mode: str = "train"):
        self.init(mode)
        for obj in self.values():
            if hasattr(obj, "start") and callable(obj.start):
                obj.start(self)

    def state_dict(self):
        state_dict = {}
        for name, obj in self.items():
            if hasattr(obj, "state_dict") and callable(obj.state_dict):
                state_dict[name] = obj.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for name in state_dict:
            self[name].load_state_dict(state_dict[name])

    def save(self, directory):
        for name, state_dict in self.state_dict().items():
            path = os.path.join(directory, name)
            if isinstance(self[name], ivory.core.state.State):
                ivory.core.state.save(state_dict, path)
            else:
                self.save_instance(state_dict, path)

    def save_instance(self, state_dict, path):
        raise NotImplementedError

    def load(self, directory):
        state_dict = {}
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if isinstance(self[name], ivory.core.state.State):
                state_dict[name] = ivory.core.state.load(path)
            else:
                state_dict[name] = self.load_instance(path)
        return state_dict

    def load_instance(self, path):
        raise NotImplementedError


class Task(Run):
    def create_run(self, args):
        run = self.experiment.create_run(args=args)
        if self.tracking:
            self.tracking.set_parent_run_id(run.id, self.id)
        return run

    def product(self, args=None, repeat=1, **kwargs):
        for args in tqdm(list(parser.product(args, **kwargs))):
            run = self.experiment.create_run(args=args)
            yield run


class Study(Task):
    def optimize(self, suggest_name: str, **kwargs):
        experiment = self.experiment
        study_name = ".".join([experiment.name, suggest_name, self.name])
        mode = experiment.create_instance("run.monitor").mode
        study = self.tuner.create_study(study_name, mode)
        if experiment.id:
            study.set_user_attr("experiment_id", experiment.id)
        has_pruning = self.tuner.pruner is not None
        objective = self.objective(suggest_name, self.create_run, has_pruning)
        study.optimize(objective, **kwargs)
        return study
