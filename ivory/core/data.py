from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Data:
    def __post_init__(self):
        self.fold = None

    def __call__(self):
        """Initializes data.

        Called from ivory.core.data.DataLoader.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """Returns a subset of data according to `index`.

        Called from ivory.core.data.DataLoader.
        """
        return self.index[index], self.input[index], self.target[index]


@dataclass
class DataLoader:
    dataset: Callable
    fold: int = 0
    batch_size: int = 32

    def __post_init__(self):
        self.train_dataloader = self.val_dataloader = None

    def __call__(self, data: Data):
        if data.fold is None:
            data()
        dataset = self.get_train_dataset(data)
        self.train_dataloader = self.get_train_dataloader(dataset)
        dataset = self.get_val_dataset(data)
        self.val_dataloader = self.get_val_dataloader(dataset)

    def get_train_dataset(self, data: Data):
        index = np.arange(len(data.fold))
        index = index[data.fold != self.fold]
        return self.get_dataset("train", data, index)

    def get_val_dataset(self, data: Data):
        index = np.arange(len(data.fold))
        index = index[data.fold == self.fold]
        return self.get_dataset("val", data, index)

    def get_dataset(self, mode: str, data, index):
        return self.dataset(mode, data[index])

    def get_train_dataloader(self, dataset):
        raise NotImplementedError

    def get_val_dataloader(self, dataset):
        raise NotImplementedError
