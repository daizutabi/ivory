import lightgbm as lgb

import ivory.core.estimator
from ivory.core import instance
from ivory.core.run import Run


class Estimator(ivory.core.estimator.Estimator):
    def __init__(self, **kwargs):
        self.params, self.kwargs = instance.filter_params(lgb.train, **kwargs)

    def step(self, run: Run, mode: str):  # type:ignore
        if mode == "train":
            train_dataset = run.datasets.train[:][1:]
            val_dataset = run.datasets.val[:][1:]
            self.fit(train_dataset, val_dataset)
        super().step(run, mode, training=False)

    def fit(self, train_dataset, val_dataset):
        train_set = lgb.Dataset(*train_dataset)
        val_set = lgb.Dataset(*val_dataset)
        valid_sets = [train_set, val_set]
        self.estimator = lgb.train(
            self.params, train_set, valid_sets=valid_sets, **self.kwargs
        )


class Regressor(Estimator):
    def __init__(self, objective="regression", metric="mse", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)


class Classifier(Estimator):
    def __init__(self, objective="multiclass", metric="multi_logloss", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)
