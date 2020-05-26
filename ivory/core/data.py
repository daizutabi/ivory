from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np

import ivory.core.collections

Index = Union[int, np.ndarray]
Value = Union[np.ndarray, Dict[str, np.ndarray]]


@dataclass
class Data:
    """Base class to provide data to `Dataset`. """

    def __post_init__(self):
        self.fold = None
        self.index = None
        self.input = None
        self.target = None
        self.init()

    def __repr__(self):
        cls_name = self.__class__.__name__
        if self.fold is None:
            return f"{cls_name}()"
        else:
            num_train = self.fold[self.fold != -1].shape[0]
            num_test = len(self.fold) - num_train
            return f"{cls_name}(train_size={num_train}, test_size={num_test})"

    def init(self):
        """Initializes `fold`, `index`, `input`, `target`."""

    def get_index(self, mode: str, fold: int) -> np.ndarray:
        """Returns index according to the mode and fold for `Dataset`.

        Args:
            mode: `train`, `val`, or `test`.
            fold: fold number

        Returns:
            data index array.
        """
        index = np.arange(len(self.fold))
        if mode == "train":
            return index[(self.fold != fold) & (self.fold != -1)]
        elif mode == "val":
            return index[self.fold == fold]
        else:
            return index[self.fold == -1]

    def get_input(self, index: Index) -> Value:
        """
        Args:
            index: Index
        """
        return self.input[index]

    def get_target(self, index: Index) -> Value:
        return self.target[index]

    def get(self, index: Union[int, np.ndarray]) -> List[Value]:
        """Returns a tuple of (index, input, target) according to the index."""
        return [self.index[index], self.get_input(index), self.get_target(index)]


@dataclass
class Dataset:
    """Dataset class which implements `__len__()`, `__getitem__()`, `__iter__()`.

    Args:
        data: data from which to load the data.
        mode: `train`, `val`, or `test`.
        fold: fold number.
        transform: callable to transform the data.
    """

    data: Data
    mode: str
    fold: int
    transform: Optional[Callable] = None

    def __post_init__(self):
        self.index = self.data.get_index(self.mode, self.fold)
        if self.mode == "test":
            self.fold = -1
        self.init()

    def init(self):
        pass

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mode={self.mode!r}, num_samples={len(self)})"

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        if index == slice(None, None, None):
            index = None
        index, input, *target = self.get(index)
        if self.transform:
            input, *target = self.transform(self.mode, input, *target)
        return [index, input, *target]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def get(self, index=None):
        if index is None:
            return self.data.get(self.index)
        else:
            return self.data.get(self.index[index])

    def sample(self, n: int = 0, frac: float = 0.0):
        index, input, *target = self[:]
        if frac:
            n = int(len(index) * frac)
        idx = np.random.permutation(len(index))[:n]
        return [x[idx] for x in [index, input, *target]]


@dataclass
class Datasets(ivory.core.collections.Dict):
    data: Data
    dataset: Callable
    fold: int

    def __post_init__(self):
        super().__post_init__()
        for mode in ["train", "val", "test"]:
            self[mode] = self.dataset(self.data, mode, self.fold)


@dataclass
class DataLoaders(Datasets):
    batch_size: int

    def __post_init__(self):
        super().__post_init__()
        for mode in ["train", "val", "test"]:
            self[mode] = self.get_dataloader(self[mode], mode)

    def get_dataloader(self, dataset, mode):
        raise NotImplementedError
