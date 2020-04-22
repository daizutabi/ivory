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
            if self.verbose:
                tqdm.write(f"[{run.name}] {run.metrics}")
            run.on_fit_end()

    def test(self, run):
        run.on_test_start()
        self.step(run, "test")
        run.on_test_end()

    def step(self, run, mode):
        index, input, *target = run.dataloaders[mode].dataset.get()
        if mode == "train":
            self.fit(input, *target)
        output = self.transform(input)
        run.results.step(index, output, *target)
        if mode != "test":
            run.metrics.step(output, *target)
