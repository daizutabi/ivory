from ivory.core.run import Run
from ivory.core.state import State
from ivory.utils.tqdm import tqdm


class Estimator(State):
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

    def step(self, run: Run, mode: str, training: bool = True, predict=None):
        index, input, *target = run.datasets[mode][:]
        if mode == "train" and training:
            self.fit(input, *target)
        predict = predict or self.predict
        output = predict(input)
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
