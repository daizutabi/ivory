from typing import Any, Dict

import ivory
from ivory.callbacks import CallbackCaller
from ivory.core.instance import instantiate


class Run(CallbackCaller):
    def __init__(self, params: Dict[str, Any], default: Dict[str, Any] = None):
        self.name = None
        self.params = params
        objects = instantiate(params, default=default)
        for key in objects:
            setattr(self, key, objects[key])

    def __len__(self):
        return len(self.params)

    def __contains__(self, key):
        return key in self.params

    def __iter__(self):
        return iter(self.params)

    def __getitem__(self, key):
        return getattr(self, key)

    def start(self):
        self.on_fit_start()
        try:
            self.trainer.fit(self)
        finally:
            self.on_fit_end()

    def state_dict(self):
        return {x: self[x].state_dict() for x in self if hasattr(self[x], "state_dict")}

    def load_state_dict(self, state_dict):
        for x in state_dict:
            self[x].load_state_dict(state_dict[x])

    def save(self, directory):
        raise NotImplementedError

    def load(self, directory):
        raise NotImplementedError


def create_run(update: Dict[str, Any] = None, experiment=None) -> Run:
    """Create a run for an optinal update params."""
    experiment = experiment or ivory.active_experiment
    if experiment is None:
        raise ValueError("active experiment does not exist.")

    run = experiment.create_run(update)
    ivory.active_run = run
    return run
