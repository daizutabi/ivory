import inspect
from typing import Callable, Optional

from ivory.core.run import Run
from ivory.core.state import State
from ivory.utils.tqdm import tqdm


class Estimator(State):
    def __init__(self, estimator_factory: Optional[Callable] = None, **kwargs):
        self.estimator_factory = estimator_factory
        self.params = {}
        self.kwargs = {}
        if self.estimator_factory:
            keys = inspect.signature(self.estimator_factory).parameters.keys()
            for key, value in kwargs.items():
                if key in keys:
                    self.kwargs[key] = value
                else:
                    self.params[key] = value
        else:
            self.kwargs.update(kwargs)

    def __repr__(self):
        class_name = self.__class__.__name__
        kwargs = ", ".join(f"{key}={value!r}" for key, value in self.kwargs.items())
        if self.params and self.kwargs:
            return f"{class_name}(params={self.params}, {kwargs})"
        elif self.params:
            return f"{class_name}(params={self.params})"
        elif self.kwargs:
            return f"{class_name}({kwargs})"
        else:
            return f"{class_name}()"

    def start(self, run: Run):
        if run.mode == "train":
            self.train(run)
        else:
            self.test(run)

    def train(self, run: Run):
        run.on_fit_begin()
        run.on_epoch_begin()
        run.on_train_begin()
        self.step(run, "train")
        run.on_train_end()
        run.on_val_begin()
        self.step(run, "val")
        run.on_val_end()
        run.on_epoch_end()
        self.log(run)
        run.on_fit_end()

    def test(self, run: Run):
        run.on_test_begin()
        self.step(run, "test")
        run.on_test_end()

    def step(self, run: Run, mode: str, training: bool = True):
        index, input, *target = run.datasets[mode][:]
        if mode == "train" and training:
            self.fit(input, *target)
        output = self.predict(input)
        if run.results:
            run.results.step(index, output, *target)

    def fit(self, input, target):
        self.estimator.fit(input, target)

    def predict(self, input):
        return self.estimator.predict(input)

    def log(self, run: Run):
        if run.metrics:
            metrics = str(run.metrics)
            if metrics:
                tqdm.write(f"[{run.name}] {metrics}")
