import lightgbm as lgb
import optuna.integration.lightgbm as lgb_tuner

import ivory.core.estimator
from ivory.core.run import Run


class Estimator(ivory.core.estimator.Estimator):
    __estimator__ = lgb.train

    def fit(self, input, target):
        train_set = lgb.Dataset(input[0], target[0])
        val_set = lgb.Dataset(input[1], target[1])
        valid_sets = [train_set, val_set]
        self.estimator = self.__estimator__(
            self.params, train_set, valid_sets=valid_sets, **self.kwargs
        )

    def step(self, run: Run, mode: str):
        if mode == "train":
            _, train_input, train_target = run.datasets.train.get()
            _, val_input, val_target = run.datasets.val.get()
            self.fit([train_input, val_input], [train_target, val_target])
        index, input, *target = run.datasets[mode].get()
        output = self.predict(input)
        run.results.step(index, output, *target)
        if mode != "test" and run.metrics:
            run.metrics.step(input, output, *target)


class Regressor(Estimator):
    def __init__(self, objective="regression", metric="rmse", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)


class Classifier(Estimator):
    def __init__(self, objective="multiclass", metric="multi_logloss", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)


class TuningEstimator(Estimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_params = {}
        self.tuning_history = []
        self.kwargs.update(
            best_params=self.best_params, tuning_history=self.tuning_history
        )
        self.__estimator__ = lgb_tuner.train


class TuningRegressor(TuningEstimator):
    def __init__(self, objective="regression", metric="rmse", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)


class TuningClassifier(TuningEstimator):
    def __init__(self, objective="multiclass", metric="multi_logloss", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)
