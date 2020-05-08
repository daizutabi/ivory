import optuna.integration.lightgbm as lgb_tuner

import ivory.core.estimator
import lightgbm as lgb
from ivory.core.run import Run


class Estimator(ivory.core.estimator.Estimator):
    def __init__(self, **kwargs):
        super().__init__(lgb.train, **kwargs)

    def set_tuner(self):
        self.estimator_factory = lgb_tuner.train
        self.best_params = {}
        self.tuning_history = []
        self.kwargs.update(
            best_params=self.best_params, tuning_history=self.tuning_history
        )

    def fit(self, input, target, val=None):
        train_set = lgb.Dataset(input, target)
        if val is not None:
            val_set = lgb.Dataset(*val)
            valid_sets = [train_set, val_set]
        else:
            valid_sets = [train_set]
        self.estimator = self.estimator_factory(
            self.params, train_set, valid_sets=valid_sets, **self.kwargs
        )

    def step(self, run: Run, mode: str):
        if mode == "train":
            _, train_input, train_target = run.datasets.train.get()
            _, val_input, val_target = run.datasets.val.get()
            self.fit(train_input, train_target, [val_input, val_target])
        index, input, *target = run.datasets[mode].get()
        output = self.predict(input)
        if run.results:
            run.results.step(index, output, *target)
        if mode != "test" and run.metrics:
            run.metrics.step(input, output, *target)


class Regressor(Estimator):
    def __init__(self, objective="regression", metric="rmse", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)


class Classifier(Estimator):
    def __init__(self, objective="multiclass", metric="multi_logloss", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)
