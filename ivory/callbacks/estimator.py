from dataclasses import dataclass

from tqdm import tqdm

from ivory.core.state import State


@dataclass
class Estimator(State):
    verbose: int = 1

    def start(self, run):
        if run.mode == "train":
            self.train(run)
        else:
            self.test(run)

    def train(self, run):
        run.on_fit_start()
        try:
            self.step(run)
        finally:
            run.on_fit_end()

    def test(self, run):
        self.test_step(run)

    def step(self, run):
        self.train_step(run)
        self.val_step(run)
        if self.verbose:
            tqdm.write(f"[{run.name}] {run.metrics}")

    def train_step(self, run):
        run.on_train_start()
        index, input, target = run.dataloaders.train.dataset.get()
        run.model.fit(input, target)
        output = run.model.transform(input)
        run.results.step(index, output, target)
        run.metrics.step(output, target)
        run.on_train_end()

    def val_step(self, run):
        run.on_val_start()
        index, input, target = run.dataloaders.val.dataset.get()
        output = run.model.transform(input)
        run.results.step(index, output, target)
        run.metrics.step(output, target)
        run.on_val_end()
        run.on_epoch_end()

    def test_step(self, run):
        run.on_test_start()
        index, input = run.dataloaders.test.dataset.get()
        output = run.model.transform(input)
        run.results.step(index, output)
        run.on_test_end()
