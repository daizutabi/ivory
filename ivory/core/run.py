import functools
import gc
import inspect
import os
import warnings
from typing import Any, Dict

from termcolor import colored

import ivory.core.collections
import ivory.core.state
from ivory import utils
from ivory.core.base import CallbackCaller
from ivory.utils.tqdm import tqdm


class Run(CallbackCaller):
    def set_tracker(self, tracker, name: str):
        if not self.id:
            self.name, self.id = tracker.create_run(
                self.experiment_id, name, self.source_name
            )
            self.params[name]["name"] = self.name
            self.params[name]["id"] = self.id
        self.set(tracker=tracker)
        self.set(tracking=tracker.create_tracking())

    def init(self, mode: str = "train"):
        self.create_callbacks()
        self.mode = mode
        self.on_init_start()
        self.on_init_end()

    def start(self, mode: str = "train"):
        if mode == "both":
            self.start("train")
            if self.tracker:
                self.tracker.load_state_dict(self, "best")
                self.start("test")
        else:
            self.init(mode)
            for obj in self.values():
                if hasattr(obj, "start") and callable(obj.start):
                    obj.start(self)

    def state_dict(self):
        state_dict = {}
        for name, obj in self.items():
            if hasattr(obj, "state_dict") and callable(obj.state_dict):
                with warnings.catch_warnings():  # for torch LambdaLR scheduler
                    warnings.simplefilter("ignore")
                    state_dict[name] = obj.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name in state_dict:
            with warnings.catch_warnings():  # for torch LambdaLR scheduler
                warnings.simplefilter("ignore")
                self[name].load_state_dict(state_dict[name])

    def save(self, directory: str):
        for name, state_dict in self.state_dict().items():
            path = os.path.join(directory, name)
            if hasattr(self[name], "save") and callable(self[name].save):
                self[name].save(state_dict, path)
            elif isinstance(self[name], ivory.core.state.State):
                ivory.core.state.save(state_dict, path)
            else:
                self.save_instance(state_dict, path)

    def save_instance(self, state_dict: Dict[str, Any], path: str):
        raise NotImplementedError

    def load(self, directory: str) -> Dict[str, Any]:
        state_dict = {}
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if hasattr(self[name], "load") and callable(self[name].load):
                state_dict[name] = self[name].load(path)
            elif isinstance(self[name], ivory.core.state.State):
                state_dict[name] = ivory.core.state.load(path)
            else:
                state_dict[name] = self.load_instance(path)
        return state_dict

    def load_instance(self, path):
        raise NotImplementedError


class Task(Run):
    def create_run(self, args, **kwargs):
        run = super().create_run(args, **kwargs)
        run_name = colored(f"[{run.name}]", "green")
        msg = utils.params.to_str(args)
        tqdm.write(run_name + f" {msg}")
        if self.tracking:
            self.tracking.set_parent_run_id(run.id, self.id)
        return run

    def terminate(self):
        if self.tracking:
            self.tracking.set_terminated(self.id)

    def product(self, params: Dict[str, Any], repeat: int = 1):
        if self.tracking:
            self.tracking.set_tags(self.id, params)
        params_list = list(utils.params.product(params)) * repeat
        try:
            for args in tqdm(params_list, desc="Run  "):
                run = self.create_run(args)
                yield run
                del run
                gc.collect()
        finally:
            self.terminate()


class Study(Task):
    def optimize(self, suggest_name: str, **kwargs):
        study_name = ".".join([self.experiment_name, suggest_name, self.name])
        mode = self.create_instance("monitor").mode
        study = self.tuner.create_study(study_name, mode)
        if self.tracking:
            self.tracking.set_tags(self.id, {"study_name": study_name})
            study.set_user_attr("run_id", self.id)
        has_pruning = self.tuner.pruner is not None
        keys = inspect.signature(study.optimize).parameters.keys()
        params = {}
        for key in list(kwargs.keys()):
            if key not in keys:
                params[key] = kwargs.pop(key)
        create_run = functools.partial(self.create_run, **params)
        objective = self.objective(suggest_name, create_run, has_pruning)
        try:
            study.optimize(objective, **kwargs)
        finally:
            self.terminate()
        return study

    def optimize_params(self, params: Dict[str, Any], **kwargs):
        if self.tracking:
            self.tracking.set_tags(self.id, params)
        suggest_name = self.objective.create_suggest(params)
        self.optimize(suggest_name, **kwargs)
