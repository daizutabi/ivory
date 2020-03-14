import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame

from ivory.torch.data import DataLoaders
from ivory.utils import kfold_split


def load_data():
    num_samples = 1000
    xy = 4 * np.random.rand(num_samples, 2) + 1
    xy = xy.astype(np.float32)
    df = DataFrame(xy, columns=["x", "y"])
    dx = 0.1 * (np.random.rand(num_samples) - 0.5)
    dy = 0.1 * (np.random.rand(num_samples) - 0.5)
    df["z"] = ((df.x + dx) * (df.y + dy)).astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df


def dataloaders():
    data = load_data()
    return DataLoaders(
        data[["x", "y"]],
        data[["z", "fold"]],
        batch_size=10,
        dataset_class="ivory.torch.Dataset",
    )


class Model(nn.Module):
    def __init__(self, hidden_sizes=[10]):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        layers = []
        for in_features, out_features in zip([2] + hidden_sizes, hidden_sizes + [1]):
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def objective(trial):
    trial.suggest_loguniform("optimizer.lr", 1e-5, 1e-1)
