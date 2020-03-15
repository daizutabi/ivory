from ivory.core.base import CallbackCaller


class Run(CallbackCaller):
    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"{class_name}(id='{self.id}', name='{self.name}', num_objects={len(self)})"
        return s

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
