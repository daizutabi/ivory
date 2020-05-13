"""A container to store training, validation and test results. """
from ivory.core.collections import Dict
from ivory.core.run import Run
from ivory.core.state import State


class Results(Dict, State):
    def reset(self):
        self.index = None
        self.output = None
        self.target = None

    def on_train_begin(self, run: Run):
        self.reset()

    def on_test_begin(self, run: Run):
        self.reset()

    def step(self, index, output, target=None):
        self.index = index
        self.output = output
        self.target = target

    def on_train_end(self, run: Run):
        self["train"] = self.result_dict()
        self.reset()

    def on_val_end(self, run: Run):
        self["val"] = self.result_dict()
        self.reset()

    def on_test_end(self, run: Run):
        self["test"] = self.result_dict()
        self.reset()

    def result_dict(self):
        return dict(index=self.index, output=self.output, target=self.target)
