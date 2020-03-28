import os

import torch

import ivory.core.run


class Run(ivory.core.run.Run):
    def save_instance(self, state_dict, directory, x):
        path = os.path.join(directory, f"{x}.pt")
        torch.save(state_dict, path)

    def load_instance(self, directory, x):
        path = os.path.join(directory, f"{x}.pt")
        return torch.load(path)
