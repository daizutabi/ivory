import pytest

from ivory.core.exceptions import EarlyStopped


def test_early_stopping(run):
    early_stopping = run.early_stopping
    early_stopping.wait = 1
    run.monitor.is_best = True
    early_stopping.on_epoch_end(run)
    assert early_stopping.wait == 0

    run.monitor.is_best = False
    early_stopping.wait = 0
    for _ in range(early_stopping.patience - 1):
        early_stopping.on_epoch_end(run)
    with pytest.raises(EarlyStopped):
        early_stopping.on_epoch_end(run)
