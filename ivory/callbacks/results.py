"""A container to store training, validation and test results. """
from typing import Dict, Iterable

import numpy as np

import ivory.core.collections
from ivory.core.run import Run
from ivory.core.state import State


class Results(ivory.core.collections.Dict, State):
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


def concatenate(iterable: Iterable[Results], callback=None) -> Results:
    indexes: Dict[str, list] = {"val": [], "test": []}
    outputs: Dict[str, list] = {"val": [], "test": []}
    targets: Dict[str, list] = {"val": [], "test": []}
    for results in iterable:
        for mode in ["val", "test"]:
            if mode not in results:
                continue
            result = results[mode]
            index, output, target = result["index"], result["output"], result["target"]
            if callback:
                index, output, target = callback(index, output, target)
            indexes[mode].append(index)
            outputs[mode].append(output)
            targets[mode].append(target)
    results = Results()
    for mode in ["val", "test"]:
        index = np.concatenate(indexes[mode])
        output = np.concatenate(outputs[mode])
        target = np.concatenate(targets[mode])
        results[mode] = dict(index=index, output=output, target=target)
    return results
