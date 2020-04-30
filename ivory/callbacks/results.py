import numpy as np
import pandas as pd

from ivory.core.collections import Dict
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

    def to_dataframe(self):
        def to_dataframe(result):
            index = result["index"].reshape(-1)
            output = result["output"]
            target = result["target"]
            if output.ndim == 3:
                num_classes = output.shape[1]
                output = output.transpose(0, 2, 1).reshape(-1, num_classes)
            output = pd.DataFrame(output, index=index)
            if target is not None:
                target = target.reshape(len(index), -1)
                if target.shape[1] == 1:
                    target = target.reshape(-1)
                if target.ndim == 1:
                    target = pd.Series(target, index=index)
                else:
                    target = pd.DataFrame(target, index=index)
            return output, target

        val_output, val_target = to_dataframe(self.val)
        test_output, test_target = to_dataframe(self.test)
        output = pd.concat([val_output, test_output])
        if test_target is None:
            target = val_target
        else:
            target = pd.concat([val_target, test_target])
        return output, target
