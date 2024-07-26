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


def test_monitor_repr():
    monitor = Monitor(mode="min")
    assert "mode='min'" in repr(monitor)
    monitor.best_score = 0.1
    monitor.best_epoch = 10
    assert "best_score=0.1, best_epoch=10" in repr(monitor)


def test_mode(run):
    run.start()
    monitor = run.monitor
    mode = monitor.mode
    monitor.mode = "max"
    monitor.best_score = -1e10
    monitor.on_epoch_end(run)
    assert monitor.is_best
    monitor.mode = mode
