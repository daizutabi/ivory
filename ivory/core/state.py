import pickle
from typing import Any, Dict


class State:
    def state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        for key, value in self.__dict__.items():
            if not callable(value):
                state_dict[key] = value
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.__dict__.update(state_dict)


def save(state_dict: Dict[str, Any], path: str):
    with open(path, "wb") as file:
        pickle.dump(state_dict, file)


def load(path: str) -> Dict[str, Any]:
    with open(path, "rb") as file:
        return pickle.load(file)
