import lightgbm as lgb

import ivory.core.estimator
from ivory.core.run import Run


class Estimator(ivory.core.estimator.Estimator):
    __estimator__ = lgb.train

    def step(self, run: Run, mode: str):
        if mode == "train":
            index, input, target = run.datasets["val"].get()
            val_set = lgb.Dataset(input, target)
            index, input, target = run.datasets["train"].get()
            train_set = lgb.Dataset(input, target)
            self.estimator = lgb.train(
                self.params, train_set, valid_sets=[val_set], **self.kwargs
            )
            target = [target]
        else:
            index, input, *target = run.datasets[mode].get()
        output = self.predict(input)
        run.results.step(index, output, *target)
        if mode != "test" and run.metrics:
            run.metrics.step(input, output, *target)


class Regressor(Estimator):
    def init(self, objective="regression", metric="rmse", **kwargs):
        super().__init__(objective=objective, metric=metric, **kwargs)


class Classifier(Estimator):
    def init(self, objective="multiclass", **kwargs):
        super().__init__(objective=objective, **kwargs)
