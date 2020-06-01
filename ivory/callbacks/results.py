"""A container to store training, validation and test results. """
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

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
        dict = ivory.core.collections.Dict()
        return dict(index=self.index, output=self.output, target=self.target)

    # def set(self, **kwargs):
    #     results = {}
    #     for key, value in kwargs.items():
    #         dict = ivory.core.collections.Dict()
    #         if len(value) == 3:
    #             dict(index=value[0], output=value[1], target=value[2])
    #         else:
    #             dict(index=value[0], output=value[1], target=None)
    #         results[key] = dict
    #     super().set(**results)

    def mean(self):
        results = Results()
        for mode, result in self.items():
            index = result.index
            kwargs = {}
            for key, value in list(result.items())[1:]:
                if value.ndim == 1:
                    series = pd.Series(value, index=index)
                    value = series.groupby(level=0).mean()
                else:
                    df = pd.DataFrame(value)
                    df["index"] = index
                    value = df.groupby("index").mean()
                value.sort_index(inplace=True)
                kwargs[key] = value.to_numpy()
                kwargs["index"] = value.index.to_numpy()
            dict = ivory.core.collections.Dict()
            results[mode] = dict(**kwargs)
        return results


class BatchResults(Results):
    def reset(self):
        super().reset()
        self.indexes = []
        self.outputs = []
        self.targets = []

    def step(self, index, output, target=None):
        self.indexes.append(index)
        self.outputs.append(output)
        if target is not None:
            self.targets.append(target)

    def result_dict(self):
        index = np.concatenate(self.indexes)
        output = np.concatenate(self.outputs)
        if self.targets:
            target = np.concatenate(self.targets)
        else:
            target = None
        super().step(index, output, target)
        return super().result_dict()


# def stack(x: List[np.ndarray]) -> np.ndarray:
#     if x[0].ndim == 1:
#         return np.hstack(x)
#     else:
#         return np.vstack(x)


def concatenate(
    iterable: Iterable[Results],
    callback: Optional[Callable] = None,
    modes: Iterable[str] = ("val", "test"),
    reduction: str = "none",
) -> Results:
    modes = list(modes)
    indexes: Dict[str, list] = {mode: [] for mode in modes}
    outputs: Dict[str, list] = {mode: [] for mode in modes}
    targets: Dict[str, list] = {mode: [] for mode in modes}
    for results in iterable:
        for mode in modes:
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
    for mode in modes:
        index = np.concatenate(indexes[mode])
        output = np.concatenate(outputs[mode])
        target = np.concatenate(targets[mode])
        dict = ivory.core.collections.Dict()
        results[mode] = dict(index=index, output=output, target=target)
    if reduction != "none":
        results = getattr(results, reduction)()
    return results
