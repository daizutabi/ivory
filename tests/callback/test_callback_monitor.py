import numpy as np
import pytest

from ivory.callback.monitor import Monitor


def test_monitor():
    monitor = Monitor(mode="min")
    assert monitor.best_score is np.inf
    monitor = Monitor(mode="max")
    assert monitor.best_score < -1e10
    with pytest.raises(ValueError):
        Monitor(mode="mean")
