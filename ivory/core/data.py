from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

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
class Dataset:
    mode: str
    data: Any
    transform: Optional[Callable[[str, Any, Any], Tuple[Any, Any]]] = None

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mode={self.mode}, num_samples={len(self)})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index, input, target = self.get(index)
        if self.transform:
            input, target = self.transform(self.mode, input, target)
        if target is None:
            return index, input
        else:
            return index, input, target

    def get(self, index) -> Tuple[Any, Any, Any]:
        """Returns a tuple of (data index, input, target)."""
        raise NotImplementedError


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
        return dataset

    def get_val_dataloader(self, dataset):
        return dataset
