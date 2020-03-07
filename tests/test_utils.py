import numpy as np

from ivory.utils import kfold_split


def test_kfold_split():
    x = np.arange(100)
    fold = kfold_split(x, 5)
    assert fold.min() == 0
    assert fold.max() == 4
    assert len(fold[fold == 3]) == 20
