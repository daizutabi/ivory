import os

import torch

import ivory


class Run(ivory.core.Run):
    def save(self, directory):
        for key, state_dict in self.state_dict().items():
            path = os.path.join(directory, f"{key}.pt")
            torch.save(state_dict, path)

    def load(self, directory):
        state_dict = {}
        for path in os.listdir(directory):
            if path.endswith(".pt"):
                state_dict[path[:-3]] = torch.load(os.path.join(directory, path))
        self.load_state_dict(state_dict)
