import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from pandas import DataFrame

from ivory.torch.data import DataFrameLoaders
from ivory.torch.metrics import Metrics
from ivory.torch.runner import Runner
from ivory.torch.trainer import Trainer
from ivory.utils import kfold_split


@pytest.fixture(scope="session")
def data():
    num_samples = 1000
    xy = 4 * np.random.rand(num_samples, 2) + 1
    xy = xy.astype(np.float32)
    df = DataFrame(xy, columns=["x", "y"])
    dx = 0.1 * (np.random.rand(num_samples) - 0.5)
    dy = 0.1 * (np.random.rand(num_samples) - 0.5)
    df["z"] = ((df.x + dx) * (df.y + dy)).astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df


@pytest.fixture
def dataloaders(data):
    return DataFrameLoaders(data, input=["x", "y"], target=["z"], batch_size=10)


@pytest.fixture
def metrics():
    return Metrics(criterion="torch.nn.functional.mse_loss")


@pytest.fixture
def trainer():
    return Trainer(max_epochs=5)


class Model(nn.Module):
    def __init__(self, hidden_sizes):
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


@pytest.fixture(scope="session")
def model():
    return Model(hidden_sizes=[10])


@pytest.fixture(scope="session")
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=1e-3)


@pytest.fixture(scope="session")
def scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


@pytest.fixture
def cfg(dataloaders, metrics, model, optimizer, scheduler, trainer):
    return Runner(
        dict(
            dataloaders=dataloaders,
            metrics=metrics,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer=trainer,
        )
    )
