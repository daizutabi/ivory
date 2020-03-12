import mlflow
import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from pandas import DataFrame

from ivory.callbacks import EarlyStopping, Pruning, Tracking
from ivory.torch.data import DataLoaders
from ivory.torch.metrics import Metrics
from ivory.torch.run import Run
from ivory.torch.trainer import Trainer
from ivory.utils import kfold_split


@pytest.fixture()
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
    return DataLoaders(
        data[["x", "y"]],
        data[["z", "fold"]],
        batch_size=10,
        dataset_class="ivory.torch.Dataset",
    )


@pytest.fixture
def metrics():
    return Metrics(criterion="torch.nn.functional.mse_loss")


if torch.cuda.is_available():
    gpu_amp = [(False, None), (True, None)]
else:
    gpu_amp = [(False, None)]


@pytest.fixture(params=gpu_amp)
def trainer(request):
    return Trainer(max_epochs=5, gpu=request.param[0], amp_level=request.param[1])


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


@pytest.fixture()
def model():
    return Model(hidden_sizes=[10])


@pytest.fixture()
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=1e-3)


@pytest.fixture(params=["step", "reduce"])
def scheduler(request, optimizer):
    if request.param == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


@pytest.fixture()
def early_stopping():
    return EarlyStopping(patience=100)


@pytest.fixture()
def tracking(tmpdir):
    tmpdir = str(tmpdir)
    if "\\" in tmpdir:
        tracking_uri = "file:///" + str(tmpdir).replace("\\", "/")
    else:
        tracking_uri = "file:" + str(tmpdir)
    mlflow.set_tracking_uri(tracking_uri)
    return Tracking()


class Trial:
    def __init__(self):
        self.step = 0

    def report(self, score, step):
        self.step = step

    def should_prune(self):
        return self.step > 10


@pytest.fixture()
def run(
    dataloaders, metrics, model, optimizer, scheduler, early_stopping, tracking, trainer
):
    trial = Trial()
    return Run(
        name="",
        params=dict(
            dataloaders=dataloaders,
            metrics=metrics,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            tracking=tracking,
            trainer=trainer,
        ),
        default={},
        callbacks=[Pruning(trial, "val_loss")],
    )
