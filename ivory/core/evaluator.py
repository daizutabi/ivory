from typing import Iterable, Tuple

import numpy as np
from pandas import DataFrame

import ivory.utils.data
from ivory.utils.tqdm import tqdm


class Evaluator:
    def __init__(self, client, run_ids=None):
        self.client = client
        self.run_ids = run_ids or []
        self._runs = []
        self.output = None
        self.target = None

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(num_runs={len(self.run_ids)})"

    @property
    def run_ids(self):
        return self._run_ids

    @run_ids.setter
    def run_ids(self, run_ids: Iterable[str]):
        self._run_ids = list(run_ids)


    def from_results(self, softmax=False, argmax=True, verbose: bool = True):
        output, target = self.load_results(verbose)
        if softmax:
            output = ivory.utils.data.softmax(output)
        output = ivory.utils.data.mean(output)
        target = ivory.utils.data.mean(target)
        if argmax:
            output = ivory.utils.data.argmax(output)
        self.output, self.target = output, target
        return output, target
