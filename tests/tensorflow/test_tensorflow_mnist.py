import pytest
import tensorflow as tf

from ivory.tensorflow.trainer import Trainer


def test_mnist(client):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    run = client.create_run("mnist")
    run.start("train")
    run.start("test")
    assert run.results.train["output"].shape == (48000, 10)
    assert run.results.val["output"].shape == (12000, 10)
    assert run.results.test["output"].shape == (10000, 10)

    assert repr(run.trainer) == "Trainer()"

    with pytest.raises(ValueError):
        Trainer(run.model, abc=123)
