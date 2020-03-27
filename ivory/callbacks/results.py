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
        self["train"] = self.data_dict()

    def on_val_end(self, run):
        self["val"] = self.data_dict()

    def on_test_end(self, run):
        self["test"] = self.data_dict()

    def data_dict(self):
        """Create data from validation/test data."""
        index = list_stack(self.indexes)
        output = list_stack(self.outputs)
        if not self.targets:
            return dict(index=index, output=output)
        target = list_stack(self.targets)
        return dict(index=index, output=output, target=target)


def list_stack(x):
    if not x:
        return
    elif x[0].ndim == 1:
        return np.hstack(x)
    else:
        return np.vstack(x)
