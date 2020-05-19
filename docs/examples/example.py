from dataclasses import dataclass

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import ivory.core.data
import ivory.core.estimator
import ivory.torch.data
from ivory.utils.fold import kfold_split


def create_data(num_samples=1000):
    xy = 4 * np.random.rand(num_samples, 2) + 1
    xy = xy.astype(np.float32)
    dx = 0.1 * (np.random.rand(num_samples) - 0.5)
    dy = 0.1 * (np.random.rand(num_samples) - 0.5)
    z = ((xy[:, 0] + dx) * (xy[:, 1] + dy)).astype(np.float32)
    z = z.reshape(-1, 1)
    return xy, z


@dataclass
class Data(ivory.core.data.Data):
    n_splits: int = 5

    DATA = create_data(1000)

    def init(self):
        self.input, self.target = self.DATA
        self.index = np.arange(len(self.input))
        self.fold = kfold_split(self.input, n_splits=self.n_splits)
        self.fold = np.where(self.fold == 4, -1, self.fold)


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


def suggest_lr(trial):
    trial.suggest_loguniform("lr", 1e-4, 1e-1)


def suggest_hidden_sizes(trial, max_num_layers=3):
    num_layers = trial.suggest_int("num_layers", 2, max_num_layers)
    for k in range(num_layers):
        trial.suggest_int(f"hidden_sizes:{k}", 10, 30)
