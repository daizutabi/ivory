import numpy as np


def test_mnist(client):
    run = client.create_run("mnist")
    run.start("train")
    run.start("test")
    assert run.results.train["output"].shape == (800, 10)
    assert run.results.val["output"].shape == (200, 10)
    assert run.results.test["output"].shape == (100, 10)
    assert np.allclose(run.metrics.val_acc, run.metrics.score)
    assert np.allclose(run.metrics.val_acc, run.metrics.callback_score)

    assert repr(run.trainer) == "Trainer()"
