import os

import ivory.core.state
from ivory.core.base import CallbackCaller


class Run(CallbackCaller):
    def set_experiment(self, experiment):
        if experiment.source_name:
            self.source_name = experiment.source_name
        if experiment.tracker:
            self.set_tracking(experiment.tracker, experiment.id)

    def set_tracking(self, tracker, experiment_id: str):
        if not self.id:
            self.id = tracker.create_run(experiment_id, self.name, self.source_name)
            self.params["run"]["id"] = self.id
        self["tracking"] = tracker.create_tracking()

    def init(self, mode: str = "train"):
        self.create_callback()
        self.mode = mode
        self.on_init()

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
