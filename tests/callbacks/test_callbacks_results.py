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
    results.step([1, 2], [3, 4], [5, 6])
    results.step([1, 2], [3, 4], [5, 6])
    results.on_val_end(None)
    results.on_test_start(None)
    results.step([1, 2], [3, 4])
    assert results.indexes == [[1, 2]]
    assert results.outputs == [[3, 4]]
    assert results.targets == []
    results.step([1, 2], [3, 4])
    results.on_test_end(None)
    assert results.train is None
    assert np.allclose(results.val["target"], [[5, 6], [5, 6]])
    assert np.allclose(results.test["index"], [[1, 2], [1, 2]])
