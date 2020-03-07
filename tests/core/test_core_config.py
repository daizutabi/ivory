import numpy as np

from ivory.core.config import Config


def test_config():
    b = np.array([1, 2])
    config = Config(dict(a=1, b=b))
    assert repr(config) == "Config(a=<int>, b=<ndarray>)"
    assert str(config) == "Config\n  a: int\n  b: numpy.ndarray"
