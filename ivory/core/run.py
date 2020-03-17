from ivory.core.base import CallbackCaller


class Run(CallbackCaller):
    __slots__ = []  # type:ignore

    def set_tracking(self, tracker, experiment_id, param_names=None):
        if not self.id:
            self.id = tracker.create_run(self.name, experiment_id)
            self.params["id"] = self.id
        self.objects["tracking"] = tracker.create_tracking(experiment_id, param_names)

    def start(self):
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
        raise NotImplementedError

    def load(self, directory):
        raise NotImplementedError
