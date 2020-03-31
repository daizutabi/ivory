import pickle


class State:
    def state_dict(self):
        state_dict = {}
        for key, value in self.__dict__.items():
            if not callable(value):
                state_dict[key] = value
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def save(state_dict, path):
    with open(path, "wb") as file:
        pickle.dump(state_dict, file)


def load(path):
    with open(path, "rb") as file:
        return pickle.load(file)
