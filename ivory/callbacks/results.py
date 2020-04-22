import numpy as np

from ivory.core.dict import Dict
from ivory.core.state import State


class Results(Dict, State):
    def reset(self):
        self.index = None
        self.output = None
        self.target = None
        self.indexes = []
        self.outputs = []
        self.targets = []

    def on_train_start(self, run):
        self.reset()

    def on_test_start(self, run):
        self.reset()

    def step(self, index, output, target=None):
        self.index = index
        self.output = output
        self.target = target

    def on_train_end(self, run):
        self["train"] = self.result_dict()
        self.reset()

    def on_val_end(self, run):
        self["val"] = self.result_dict()
        self.reset()

    def on_test_end(self, run):
        self["test"] = self.result_dict()
        self.reset()

    def result_dict(self):
        self.stack()
        return dict(index=self.index, output=self.output, target=self.target)

    def stack(self):
        if self.index is not None or not self.indexes:
            return
        self.index = np.vstack(self.indexes)
        self.output = np.vstack(self.outputs)
        if self.targets:
            self.target = np.vstack(self.targets)
