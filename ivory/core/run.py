import os
import pickle

from ivory.core.base import CallbackCaller


class Run(CallbackCaller):
    __slots__ = []  # type:ignore

    def set_experiment(self, experiment):
        if experiment.data:
            self.set_data(experiment.data)
        if experiment.tracker:
            self.set_tracking(experiment.tracker, experiment.id)

    def set_data(self, data):
        self.set(data=data)

    def set_tracking(self, tracker, experiment_id):
        if not self.id:
            self.id = tracker.create_run(experiment_id, self.name, self.source_name)
            self.params["run"]["id"] = self.id
        tracking = tracker.create_tracking(experiment_id)
        self.set(tracking=tracking)

    def start(self):
        self.dataloader.init(self.data)
        self.create_callbacks()
        self.trainer.fit(self)

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
        for key, state_dict in self.state_dict().items():
            path = os.path.join(directory, f"{key}.pickle")
            with open(path, "wb") as file:
                pickle.dump(state_dict, file)

    def load(self, directory):
        state_dict = {}
        for path in os.listdir(directory):
            if path.endswith(".pickle"):
                name = path.split(".")[0]
                path = os.path.join(directory, path)
                with open(path, "rb") as file:
                    state_dict[name] = pickle.load(file)
        return state_dict
