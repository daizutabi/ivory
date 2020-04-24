from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import ivory.core.dict


@dataclass
class Data:
    def __post_init__(self):
        self.fold = None
        self.index = None
        self.input = None
        self.target = None
        self.init()

    def init(self):
        pass

    def get_index(self, mode: str, fold: int):
        index = np.arange(len(self.fold))
        if mode == "train":
            return index[(self.fold != fold) & (self.fold != -1)]
        elif mode == "val":
            return index[self.fold == fold]
        else:
            return index[self.fold == -1]

    def get(self, index):
        return [self.index[index], self.input[index], self.target[index]]


@dataclass
class Dataset:
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


@dataclass
class Datasets(ivory.core.dict.Dict):
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
