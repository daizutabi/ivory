import mlflow

from ivory.callbacks import Tracking


def test_start(tmpdir):
    tmpdir = str(tmpdir)
    if "\\" in tmpdir:
        tracking_uri = "file:///" + str(tmpdir).replace("\\", "/")
    else:
        tracking_uri = "file:" + str(tmpdir)
    mlflow.set_tracking_uri(tracking_uri)
    return Tracking()


def test_experiment(experiment):
    experiment.start()
    assert Tracking.experiment_id == "0"
    experiment.start()
    assert Tracking.experiment_id == "0"
