from typing import Any, Dict

import ivory
from ivory.callbacks import CallbackCaller
from ivory.core import instance


class Run(CallbackCaller):
    __slots__ = ["name", "params", "objects"]

    def __init__(self, name, params, default=None, callbacks=None):
        self.name = name
        self.params = params
        self.objects = instance.instantiate(self.params, default=default)
        super().__init__(callbacks)

    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"{class_name}(name='{self.name}', num_objects={len(self.objects)}, "
        s += f"num_callbacks={len(self.callbacks)})"
        return s

    def __len__(self):
        return len(self.objects)

    def __contains__(self, key):
        return key in self.objects

    def __iter__(self):
        return iter(self.objects)

    def __getitem__(self, key):
        return self.objects[key]

    def __getattr__(self, key):
        if key in self.objects:
            return self.objects[key]
        else:
            return self.callbacks[key]

    def start(self):
        self.on_fit_start()
        try:
            self.trainer.fit(self)
        finally:
            self.on_fit_end()

    def state_dict(self):
        state_dict = {}
        for x in self:
            if hasattr(self[x], "state_dict"):
                state_dict[x] = self[x].state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for x in state_dict:
            if x in self:
                self[x].load_state_dict(state_dict[x])

    def save(self, directory):
        raise NotImplementedError

    def load(self, directory):
        raise NotImplementedError


def create_run(update: Dict[str, Any] = None, experiment=None, callbacks=None) -> Run:
    """Create a run for an optinal update params."""
    if experiment is None:
        experiment = ivory.active_experiment
    if experiment is None:
        raise ValueError("active experiment does not exist.")
    run = experiment.create_run(update, callbacks=callbacks)
    return run
