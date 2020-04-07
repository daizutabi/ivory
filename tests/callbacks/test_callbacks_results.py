import numpy as np

from ivory.callbacks.results import Results


def test_resutls():
    results = Results()

    results.on_train_start(None)
    assert results.indexes == []
    assert results.outputs == []
    assert results.targets == []
    results.on_train_end(None)
    results.on_val_start(None)
    results.step([1, 2], [[3, 4], [5, 6]], [10, 20])
    results.step([1, 2], [[3, 4], [5, 6]], [10, 20])
    results.on_val_end(None)
    results.on_test_start(None)
    results.step([3, 4], [[3, 4], [5, 6]])
    assert results.indexes == [[3, 4]]
    assert results.outputs == [[[3, 4], [5, 6]]]
    assert results.targets == []
    results.step([5, 6], [[3, 4], [5, 6]])
    results.on_test_end(None)
    assert results.train is None
    assert np.allclose(results.val["target"], [[10, 20], [10, 20]])
    assert np.allclose(results.test["index"], [3, 4, 5, 6])
