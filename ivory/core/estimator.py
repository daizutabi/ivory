import inspect

from ivory.core.run import Run
from ivory.core.state import State
from ivory.utils.tqdm import tqdm


class Estimator(State):
    __estimator__ = None

    def __init__(self, **kwargs):
        self.estimator = None
        self.params = {}
        self.kwargs = {}
        keys = inspect.signature(self.__estimator__).parameters.keys()
        for key, value in kwargs.items():
            if key in keys:
                self.kwargs[key] = value
            else:
                self.params[key] = value

    def fit(self, input, target):
        self.estimator.fit(input, target)

    def predict(self, input):
        return self.estimator.predict(input)

    def start(self, run: Run):
        if run.mode == "train":
            self.train(run)
        else:
            self.test(run)

    def train(self, run: Run):
        run.on_fit_start()
        run.on_epoch_start()
        run.on_train_start()
        self.step(run, "train")
        run.on_train_end()
        run.on_val_start()
        self.step(run, "val")
        run.on_val_end()
        try:
            run.on_epoch_end()
        finally:
            if run.metrics:
                tqdm.write(f"[{run.name}] {run.metrics}")
            run.on_fit_end()

    def test(self, run: Run):
        run.on_test_start()
        self.step(run, "test")
        run.on_test_end()

    def step(self, run: Run, mode: str):
        index, input, *target = run.datasets[mode].get()
        if mode == "train":
            self.fit(input, *target)
        output = self.predict(input)
        run.results.step(index, output, *target)
        if mode != "test" and run.metrics:
            run.metrics.step(input, output, *target)
