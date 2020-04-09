import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

import ivory.core.dict


@dataclass
class Data:
    def __post_init__(self):
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            self.init()
            self.initialized = True

    def init(self):
        """Initializes the data. For example, read a csv file as a DataFrame.

        Called from ivory.core.data.DataLoaders.
        """
        raise NotImplementedError

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
        index, input, *target = self.get(index)
        if self.transform:
            input, *target = self.transform(self.mode, input, *target)
        return [index, input, *target]

    def get(self, index):
        """Returns a tuple of (index, input, target) or (index, input)."""
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
    dataset: Callable
    fold: int = 0
    batch_size: int = 1

    def __repr__(self):
        cls_name = self.__class__.__name__
        if isinstance(self.dataset, functools.partial):
            dataset = self.dataset.func.__module__
            dataset += "." + self.dataset.func.__name__
            items = self.dataset.keywords.items()
            kwargs = [f"{key}={value!r}" for key, value in items]
            kwargs = ", ".join(kwargs)
        else:
            dataset = self.dataset.__module__
            dataset += "." + self.dataset.__name__
            kwargs = ""
        args = ""
        for key in self.__dataclass_fields__:
            if key != "dataset":
                args += f", {key}={getattr(self, key)!r}"
        return f"{cls_name}(dataset={dataset}({kwargs}){args})"

    def init(self, mode: str, data: Data):
        data.initialize()
        self.preprocess(data)
        if mode == "train":
            for mode in ["train", "val"]:
                index = self.get_index(mode, data)
                dataset = self.dataset(mode, data.get(mode, index))
                self[mode] = self.get_dataloader(mode, dataset)
        elif mode == "test":
            index = self.get_index("test", data)
            dataset = self.dataset("test", data.get(mode, index))
            self["test"] = self.get_dataloader("test", dataset)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def preprocess(self, data: Data):
        pass

    def get_index(self, mode, data):
        if mode == "train":
            return (data.fold != self.fold) & (data.fold != -1)
        elif mode == "val":
            return data.fold == self.fold
        elif mode == "test" and -1 in data.fold:
            return data.fold == -1
        else:
            return np.full(len(data.fold), True)

    def get_dataloader(self, mode, dataset):
        return DataLoader(dataset, batch_size=self.batch_size)
