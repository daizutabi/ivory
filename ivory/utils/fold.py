import numpy as np
from sklearn.model_selection import KFold


def fold_array(splitter, x, y=None, groups=None):
    fold = np.full(x.shape[0], -1, dtype=np.int8)
    for i, (_, test_index) in enumerate(splitter.split(x, y, groups)):
        fold[test_index] = i
    return fold


def kfold_split(x, n_splits=5, random_state=0, shuffle=True):
    splitter = KFold(n_splits, random_state=random_state, shuffle=shuffle)
    return fold_array(splitter, x)


def multilabel_stratified_kfold_split(labels, n_splits, shuffle=True, random_state=0):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    splitter = MultilabelStratifiedKFold(
        n_splits, shuffle=shuffle, random_state=random_state
    )
    x = np.arange(len(labels))
    return fold_array(splitter, x, labels)
