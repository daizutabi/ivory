from dataclasses import dataclass

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import ivory.core.data
import ivory.torch.data
from ivory.utils import kfold_split


def create_data(num_samples=1000):
    xy = 4 * np.random.rand(num_samples, 2) + 1
    xy = xy.astype(np.float32)
    dx = 0.1 * (np.random.rand(num_samples) - 0.5)
    dy = 0.1 * (np.random.rand(num_samples) - 0.5)
    z = ((xy[:, 0] + dx) * (xy[:, 1] + dy)).astype(np.float32)
    return xy, z


@dataclass
class Data(ivory.core.data.Data):
    num_samples: int = 1000

    def init(self):
        self.input, self.target = create_data(self.num_samples)
        self.index = np.arange(len(self.input))
        self.fold = kfold_split(self.input, n_splits=5)

    def get(self, index=None):
        if index is None:
            return [self.index, self.input]
        else:
            return [self.index[index], self.input[index], self.target[index]]


@dataclass(repr=False)
class Dataset(ivory.torch.data.Dataset):
    dummy: int = 10

    def __len__(self):
        return len(self.data[0])

    def get(self, index):
        return [x[index] for x in self.data]


class Model(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        for in_features, out_features in zip([2] + hidden_sizes, hidden_sizes + [1]):
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def suggest(trial):
    trial.suggest_loguniform("optimizer.lr", 1e-4, 1e-1)
