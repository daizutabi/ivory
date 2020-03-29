import os

import ivory.core.state
from ivory.core.base import CallbackCaller


class Run(CallbackCaller):
    def set_experiment(self, experiment):
        if experiment.data:
            self.set_data(experiment.data)
        if experiment.tracker:
            self.set_tracking(experiment.tracker, experiment.id)

    def set_data(self, data):
        self["data"] = data

    def set_tracking(self, tracker, experiment_id):
        if not self.id:
            self.id = tracker.create_run(experiment_id, self.name, self.source_name)
            self.params["run"]["id"] = self.id
        self["tracking"] = tracker.create_tracking()

    def start(self, mode="train", leave=True):
        if self.data.mode != mode:
            self.data.mode = mode
            self.data.initialized = False
        self.dataloaders.init(self.data)
        self.create_callback()
        if self.data.mode == "train":
            self.trainer.fit(self, leave)
        else:
            self.trainer.test(self)

    def state_dict(self):
        state_dict = {}
        for x in self:
            if hasattr(self[x], "state_dict") and callable(self[x].state_dict):
                state_dict[x] = self[x].state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for x in state_dict:
            self[x].load_state_dict(state_dict[x])

    def save(self, directory):
        for x, state_dict in self.state_dict().items():
            if isinstance(self[x], ivory.core.state.State):
                ivory.core.state.save(state_dict, directory, x)
            else:
                self.save_instance(state_dict, directory, x)

    def save_instance(self, state_dict, directory, x):
        raise ValueError(f"Unknown save method for {self[x]}.")

    def load(self, directory):
        state_dict = {}
        for path in os.listdir(directory):
            x = path.split(".")[0]
            if isinstance(self[x], ivory.core.state.State):
                state_dict[x] = ivory.core.state.load(directory, x)
            else:
                state_dict[x] = self.load_instance(directory, x)
        return state_dict

    def load_instance(self, directory, x):
        raise ValueError(f"Unknown load method for {self[x]}.")
