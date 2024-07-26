import numpy as np

from ivory.utils.fold import (kfold_split, multilabel_stratified_kfold_split,
                              stratified_kfold_split)


def test_kfold_split():
    x = np.arange(100)
    fold = kfold_split(x, 5)
    assert fold.min() == 0
    assert fold.max() == 4
    assert len(fold[fold == 3]) == 20


def test_stratified_kfold_split():
    y = np.array([0] * 16 + [1] * 8 + [2] * 4)
    fold = stratified_kfold_split(y, 4)
    assert np.allclose(y[fold == 0], [0, 0, 0, 0, 1, 1, 2])
    assert np.allclose(y[fold == 3], [0, 0, 0, 0, 1, 1, 2])


def test_multilabel_stratified_kfold_split():
    y = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0]])
    fold = multilabel_stratified_kfold_split(y, 2)
    array = fold.reshape((-1, 2))
    assert all(np.min(array, axis=1) == 0)
    assert all(np.max(array, axis=1) == 1)
