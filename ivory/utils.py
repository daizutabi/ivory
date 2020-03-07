import numpy as np
from sklearn.model_selection import KFold


def fold_array(splitter, x, y=None, groups=None) -> np.ndarray:
    fold = np.full(x.shape[0], -1, dtype=np.int8)
    for i, (_, test_index) in enumerate(splitter.split(x, y, groups)):
        fold[test_index] = i
    return fold


def kfold_split(x, n_splits=5) -> np.ndarray:
    splitter = KFold(n_splits, random_state=0, shuffle=True)
    return fold_array(splitter, x)
