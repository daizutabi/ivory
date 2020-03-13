class State:
    def state_dict(self):
        state_dict = {}
        for key, value in self.__dict__.items():
            if not callable(value):
                state_dict[key] = value
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
