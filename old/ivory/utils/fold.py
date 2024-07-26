import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def fold_array(splitter, x, y=None, groups=None):
    fold = np.full(x.shape[0], -1, dtype=np.int8)
    for i, (_, test_index) in enumerate(splitter.split(x, y, groups)):
        fold[test_index] = i
    return fold


def kfold_split(x, n_splits=5, shuffle=True, random_state=0):
    splitter = KFold(n_splits, shuffle=shuffle, random_state=random_state)
    return fold_array(splitter, x)


def group_kfold_split(groups, n_splits=5):
    splitter = GroupKFold(n_splits)
    return fold_array(splitter, groups, groups, groups=groups)


def stratified_kfold_split(y, n_splits=5, shuffle=True, random_state=0):
    splitter = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    return fold_array(splitter, y, y)


def multilabel_stratified_kfold_split(labels, n_splits, shuffle=True, random_state=0):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    splitter = MultilabelStratifiedKFold(
        n_splits, shuffle=shuffle, random_state=random_state
    )
    x = np.arange(len(labels))
    return fold_array(splitter, x, labels)
