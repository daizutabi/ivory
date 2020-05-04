from dataclasses import dataclass

from ivory.core.run import Run
from ivory.core.state import State
from ivory.utils.tqdm import tqdm


@dataclass
class Estimator(State):
    verbose: int = 1

    def fit(self, input, *target):
        raise NotImplementedError

    def transform(self, input):
        raise NotImplementedError

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
            if self.verbose and run.metrics:
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
        output = self.transform(input)
        run.results.step(index, output, *target)
        if mode != "test" and run.metrics:
            run.metrics.step(input, output, *target)
