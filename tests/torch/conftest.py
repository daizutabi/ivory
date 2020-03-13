import mlflow
import pytest
import torch

from ivory.callbacks import Pruning
from ivory.torch.run import Run
from simple import load_data


@pytest.fixture()
def data():
    return load_data()


@pytest.fixture()
def tracking_uri(tmpdir):
    tmpdir = str(tmpdir)
    if "\\" in tmpdir:
        tracking_uri = "file:///" + str(tmpdir).replace("\\", "/")
    else:
        tracking_uri = "file:" + str(tmpdir)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


class Trial:
    def __init__(self):
        self.step = 0

    def report(self, score, step):
        self.step = step

    def should_prune(self):
        return self.step > 10


if torch.cuda.is_available():
    params = [False, True]
else:
    params = [False]


@pytest.fixture(params=params)
def trainer(request):
    return {"class": "ivory.torch.Trainer", "max_epochs": 5, "gpu": request.param}


@pytest.fixture(params=["step", "reduce"])
def scheduler(request):
    if request.param == "step":
        cls = "StepLR"
        params = {"step_size": 10}
    else:
        cls = "ReduceLROnPlateau"
        params = {"patience": 10}
    return {"class": f"torch.optim.lr_scheduler.{cls}", "optimizer": "$", **params}


@pytest.fixture()
def pruning():
    return Pruning(Trial(), "val_loss")


@pytest.fixture()
def run(tracking_uri, scheduler, trainer):
    params = {
        "dataloaders": {"def": "simple.dataloaders"},
        "model": {"class": "simple.Model"},
        "optimizer": {"class": "torch.optim.Adam", "params": "$.model.parameters()"},
        "scheduler": scheduler,
        "metrics": {"class": "simple.Metrics"},
        "monitor": {"class": "ivory.callbacks.Monitor"},
        "trainer": trainer,
        "early_stopping": {"class": "ivory.callbacks.EarlyStopping", "patience": 100},
        "tracking": {"class": "ivory.callbacks.Tracking"},
    }
    callbacks = [Pruning(Trial(), "val_loss")]
    return Run("abc", params=params, callbacks=callbacks)
