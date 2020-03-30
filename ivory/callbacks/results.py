import numpy as np

from ivory.core.dict import Dict
from ivory.core.state import State


class Results(Dict, State):
    def reset(self):
        self.indexes = []
        self.outputs = []
        self.targets = []

    def on_train_start(self, run):
        self.reset()

    def on_val_start(self, run):
        self.reset()

    def on_test_start(self, run):
        self.reset()

    def step(self, index, output, target=None):
        self.indexes.append(index)
        self.outputs.append(output)
        if target is not None:
            self.targets.append(target)

    def on_train_end(self, run):
        self["train"] = self.result_dict()

    def on_val_end(self, run):
        self["val"] = self.result_dict()

    def on_test_end(self, run):
        self["test"] = self.result_dict()

    def result_dict(self):
        if not (self.indexes):
            return None
        index = np.vstack(self.indexes)
        output = np.vstack(self.outputs)
        if not self.targets:
            return dict(index=index, output=output)
        target = np.vstack(self.targets)
        return dict(index=index, output=output, target=target)
