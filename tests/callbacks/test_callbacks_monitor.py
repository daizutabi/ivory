import numpy as np
import pytest

from ivory.callbacks.monitor import Monitor


def test_monitor(run):
    monitor = Monitor(mode="min")
    assert monitor.best_score is np.inf
    monitor = Monitor(mode="max")
    assert monitor.best_score < -1e10
    with pytest.raises(ValueError):
        Monitor(mode="mean")

    monitor.metric = "abc"
    with pytest.raises(ValueError):
        monitor.on_epoch_end(run)
