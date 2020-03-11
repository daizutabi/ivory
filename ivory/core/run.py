from dataclasses import InitVar, dataclass, field
from typing import Any, Dict

import ivory
from ivory.callbacks import CallbackCaller
from ivory.core.instance import instantiate


@dataclass
class Run(CallbackCaller):
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict, repr=False)
    default: InitVar[Dict[str, Any]] = None

    def __post_init__(self, default):
        self._objects = instantiate(self.params, default=default)
        for key in self._objects:
            setattr(self, key, self._objects[key])

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(name='{self.name}', callbacks={self.callbacks})"

    def __len__(self):
        return len(self._objects)

    def __contains__(self, key):
        return key in self._objects

    def __iter__(self):
        return iter(self._objects)

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


def create_run(update: Dict[str, Any] = None, experiment=None, callbacks=None) -> Run:
    """Create a run for an optinal update params."""
    experiment = experiment or ivory.active_experiment
    if experiment is None:
        raise ValueError("active experiment does not exist.")
    run = experiment.create_run(update, callbacks=callbacks)
    return run
