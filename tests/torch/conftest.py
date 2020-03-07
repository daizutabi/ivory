import pytest

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
    return Trainer()


@pytest.fixture
def cfg(dataloaders, metrics, trainer):
    return Config(dict(dataloaders=dataloaders, metrics=metrics, trainer=trainer))
