from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

import ivory.core.dict


@dataclass
class Data:
    fold: Optional[np.ndarray] = field(default=None, repr=False)

    def init(self, dataloaders):
        pass

    def get(self, mode, index):
        """Returns a subset of data according to `mode` and `index`.

        Returned object can be any type but should be processed by Dataset's ``get()``.

        Args:
            index (list): 1d-array of bool, optional. The length is the same as `fold`.

        Called from ivory.core.data.DataLoaders.
        """
        if mode == "test":
            return [self.index[index], self.input[index]]
        else:
            return [self.index[index], self.input[index], self.target[index]]

    def get_index(self, mode: str, fold: int = -1):
        if mode == "train":
            return (self.fold != fold) & (self.fold != -1)
        elif mode == "val":
            return self.fold == fold
        elif mode == "test":
            return self.fold == -1


@dataclass
class Dataset:
    mode: str
    data: Any
    transform: Optional[Callable] = None

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mode={self.mode!r}, num_samples={len(self)})"

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        index, input, *target = self.get(index)
        if self.transform:
            input, *target = self.transform(self.mode, input, *target)
        return [index, input, *target]

    def init(self, dataloaders):
        pass

    def get(self, index=None):
        """Returns a tuple of (index, input, target) or (index, input)."""
        if index is None:
            return self.data
        else:
            return [x[index] for x in self.data]


@dataclass
class DataLoader:
    dataset: Dataset
    batch_size: int = 1

    def __post_init__(self):
        if self.batch_size != 1:
            raise NotImplementedError("batch_size muse be 1.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return [np.expand_dims(x, 0) for x in data]


@dataclass
class DataLoaders(ivory.core.dict.Dict):
    data: Data
    dataset: Callable
    fold: int = 0
    batch_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.data.init(self)
        self.init()

    def init(self):
        for mode in ["train", "val", "test"]:
            index = self.data.get_index(mode, self.fold)
            dataset = self.dataset(mode, self.data.get(mode, index))
            dataset.init(self)
            if mode == "train" and hasattr(dataset.transform, "init"):
                dataset.transform.init(dataset)
            self[mode] = self.get_dataloader(mode, dataset)

    def get_dataloader(self, mode, dataset):
        return DataLoader(dataset, batch_size=self.batch_size)
