from dataclasses import dataclass

import numpy as np

import ivory.core.data
from ivory.utils.fold import kfold_split


def create_data(num_samples=1000):
    x = np.linspace(0, 1, num_samples)
    y = np.linspace(1, 0, num_samples)
    xy = np.vstack((x, y)).T
    xy = xy.astype(np.float32)
    z = (xy[:, 0] * xy[:, 1]).astype(np.float32)
    return xy, z


@dataclass(repr=False)
class Data(ivory.core.data.Data):
    n_splits: int = 4

    DATA = create_data(1000)  # Shared by each run.

    def init(self):  # Called from self.__post_init__()
        self.input, self.target = self.DATA
        self.index = np.arange(len(self.input))
        # Extra fold for test data.
        self.fold = kfold_split(self.input, n_splits=self.n_splits + 1)

        # Creating dummy test data just for demonstration.
        is_test = self.fold == self.n_splits  # Use an extra fold.
        self.fold[is_test] = -1  # -1 for test data.
        self.target = self.target.copy()  # n_splits may be different among runs.
        self.target[is_test] = np.nan  # Delete target for test data.

        self.target = self.target.reshape(-1, 1)  # (sample, class)
