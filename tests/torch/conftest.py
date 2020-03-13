import mlflow
import pytest

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


@pytest.fixture()
def run(tracking_uri):
    params = {
        "dataloaders": {"def": "simple.dataloaders"},
        "model": {"class": "simple.Model"},
        "optimizer": {"class": "torch.optim.Adam", "params": "$.model.parameters()"},
        "metrics": {"class": "simple.Metrics"},
        "monitor": {"class": "ivory.callbacks.Monitor"},
        "trainer": {"class": "ivory.torch.Trainer", "max_epochs": 5, "gpu": False},
        "early_stopping": {"class": "ivory.callbacks.EarlyStopping", "patience": 100},
        "tracking": {"class": "ivory.callbacks.Tracking"},
    }
    callbacks = [Pruning(Trial(), "val_loss")]
    return Run("abc", params=params, callbacks=callbacks)

    run = Run("abc", params=params, callbacks=callbacks)

    run.trainer.on_fit_start

    run.on_fit_start.objects

    run.on_train_start
