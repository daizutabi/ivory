"""A container to store training, validation and test results. """
from typing import Iterable

import numpy as np

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


def concatenate(iterable: Iterable[Results], callback=None):
    indexes = []
    outputs = []
    targets = []
    for results in iterable:
        for mode in ["val", "test"]:
            if mode not in results:
                continue
            result = results[mode]
            index, output, target = result["index"], result["output"], result["target"]
            if callback:
                index, output, target = callback(index, output, target)
            indexes.append(index)
            outputs.append(output)
            targets.append(target)
    index = np.concatenate(indexes)
    output = np.concatenate(outputs)
    target = np.concatenate(targets)
    return index, output, target
