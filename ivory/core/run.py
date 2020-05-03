import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict

import ivory.core.collections
import ivory.core.state
from ivory.core import parser
from ivory.core.base import CallbackCaller
from ivory.utils.tqdm import tqdm


class Run(CallbackCaller):
    def set_tracker(self, tracker):
        if not self.id:
            experiment_id = self.experiment_id
            self.id = tracker.create_run(experiment_id, self.name, self.source_name)
            class_name = self.__class__.__name__.lower()
            self.params[class_name]["id"] = self.id
        self.set(tracker=tracker)
        self.set(tracking=tracker.create_tracking())

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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    state_dict[name] = obj.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name in state_dict:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self[name].load_state_dict(state_dict[name])

    def save(self, directory: str):
        for name, state_dict in self.state_dict().items():
            path = os.path.join(directory, name)
            if isinstance(self[name], ivory.core.state.State):
                ivory.core.state.save(state_dict, path)
            else:
                self.save_instance(state_dict, path)

    def save_instance(self, state_dict: Dict[str, Any], path: str):
        raise NotImplementedError

    def load(self, directory: str) -> Dict[str, Any]:
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


@dataclass
class Runs(ivory.core.collections.List, ivory.core.state.State):
    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(num_runs={len(self)})"


class Task(Run):
    __requires__ = ["runs"]

    def create_run(self, args):
        run = super().create_run(args)
        if self.tracking:
            self.tracking.set_parent_run_id(run.id, self.id)
            self.runs.append(run.id)
            self.tracking.log_params_artifact(self)
            self.tracking.save_run(self, "current")
        return run

    def terminate(self):
        if self.tracking:
            self.tracking.set_terminated(self.id)

    def product(self, args=None, repeat: int = 1, **kwargs):
        params = parser.parse_args(args, **kwargs)
        if self.tracking:
            self.tracking.set_tags(self.id, params)
        params = list(parser.product(params)) * repeat
        try:
            for args in tqdm(params, desc="Run  "):
                yield self.create_run(args)
        finally:
            self.terminate()


class Study(Task):
    def optimize(self, suggest_name: str, study_name: str = "", **kwargs):
        if not study_name:
            study_name = self.name
        study_name = ".".join([self.experiment_name, suggest_name, study_name])
        mode = self.create_instance("monitor").mode
        study = self.tuner.create_study(study_name, mode)
        if self.tracking:
            self.tracking.set_tags(self.id, {"study_name": study_name})
            study.set_user_attr("run_id", self.id)
        has_pruning = self.tuner.pruner is not None
        objective = self.objective(suggest_name, self.create_run, has_pruning)
        try:
            study.optimize(objective, **kwargs)
        finally:
            self.terminate()
        return study
