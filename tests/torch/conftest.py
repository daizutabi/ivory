import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from ivory.torch.data import DataFrameLoaders
from ivory.torch.metrics import Metrics
from ivory.torch.trainer import Trainer
from ivory.utils import Config


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
    return Config(
        dict(
            dataloaders=dataloaders,
            metrics=metrics,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer=trainer,
        )
    )
