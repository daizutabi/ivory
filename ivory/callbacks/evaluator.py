from dataclasses import dataclass

from tqdm import tqdm

from ivory.core.state import State


@dataclass
class Evaluator(State):
    verbose: int = 1

    def test(self, run, leave=False):
        runs = self.search_run(**self.query)
        if self.verbose == 1:
            runs = tqdm(list(runs), desc="Eval", leave=leave)
        run.on_test_start()
        for experiment, run_id in runs:
            results = experiment.load_instance(run_id, "results")
            index, output, target = self.reshape(**results.val)
            self.val_update(index, output, target)
            if results.test:
                index, output = self.reshape(**results.test)
                self.test_update(index, output)
        run.on_test_end()

    def on_val_start(self, run):
        self.val_output = []
        self.val_target = []

    def on_test_start(self, run):
        self.test_output = []
        self.test_target = []

    def on_val_end(self, run):
        self.val = self.result_dict()

    def on_test_end(self, run):
        self.test = self.result_dict()

    def result_dict(self):
        pass

    def reshape(self, index, output, target=None):
        index = index.reshape(-1)
        if output.ndim == 3:
            output = output.transpose(0, 2, 1)
            output = output.reshape((-1, output.shape[-1]))
        if target is None:
            return index, output
        if target.ndim == 3:
            target = target.transpose(0, 2, 1)
            target = target.reshape((-1, target.shape[-1]))
        return index, output, target

    def val_update(self, index, output, target):
        for i, o, t in zip(index, output, target):
            if i in self.val_output:
                self.val_output[i].append(o)
                self.val_target[i].append(t)
            else:
                self.val_output[i] = [o]
                self.val_target[i] = [t]

    def test_update(self, index, output):
        for i, o in zip(index, output):
            if i in self.test_output:
                self.test_output[i].append(o)
            else:
                self.test_output[i] = [o]
