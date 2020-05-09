import pytest

from ivory.tensorflow.trainer import Trainer


def test_mnist(client):
    run = client.create_run("mnist")
    run.start("train")
    run.start("test")
    assert run.results.train["output"].shape == (48000, 10)
    assert run.results.val["output"].shape == (12000, 10)
    assert run.results.test["output"].shape == (10000, 10)

    assert repr(run.trainer) == "Trainer()"

    with pytest.raises(ValueError):
        Trainer(run.model, abc=123)
