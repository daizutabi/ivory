import inspect
from dataclasses import dataclass

import lightgbm as lgb

import ivory.core.estimator
from ivory.core.run import Run


@dataclass
class Estimator(ivory.core.estimator.Estimator):
    num_boost_round: int = 10000
    early_stopping_rounds: int = 50
    verbose_eval: int = 100

    def __post_init__(self):
        self.params = {}
        self.kwargs = {}
        self.model = None
        keys = inspect.signature(lgb.train).parameters.keys()
        for key in self.__dataclass_fields__.keys():
            value = getattr(self, key)
            if key in keys:
                self.kwargs[key] = value
            else:
                self.params[key] = value

    def step(self, run: Run, mode: str):
        if mode == "train":
            index, input, target = run.datasets["val"].get()
            val_set = lgb.Dataset(input, target)
            index, input, target = run.datasets["train"].get()
            train_set = lgb.Dataset(input, target)
            self.model = lgb.train(
                self.params, train_set, valid_sets=[val_set], **self.kwargs
            )
            target = [target]
        else:
            index, input, *target = run.datasets[mode].get()
        output = self.model.predict(input)
        run.results.step(index, output, *target)
        if mode != "test" and run.metrics:
            run.metrics.step(input, output, *target)
