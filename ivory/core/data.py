from dataclasses import dataclass
from typing import Callable

import numpy as np
from pandas import DataFrame


@dataclass
class Data:
    data: Callable
    fold_name: str = "fold"

    def __post_init__(self):
        self.index = self.input = self.target = self.fold = None

    def __call__(self):
        if self.index is not None:
            return self
        self.input, self.target = self.data()
        if isinstance(self.input, DataFrame):
            self.input = self.to_numpy(self.input)
        if isinstance(self.target, DataFrame):
            self.target = self.to_numpy(self.target)
        if self.index is None:
            raise NotImplementedError
        return self

    def to_numpy(self, df: DataFrame):
        self.index = df.index.to_numpy()
        cs = df.columns
        if self.fold_name in cs:
            self.fold = df[self.fold_name].to_numpy()
            cs = [c for c in cs if c != self.fold_name]
        return df[cs].to_numpy()


@dataclass
class DataLoader:
    dataset: Callable
    dataloader: Callable
    fold: int = 0
    train_ratio: float = 1.0
    val_ratio: float = 1.0

    def __post_init__(self):
        self.train_dataloader = self.val_dataloader = None

    def __call__(self, data: Data):
        data()
        dataset = self.get_train_dataset(data)
        self.train_dataloader = self.dataloader("train", dataset)
        dataset = self.get_val_dataset(data)
        self.val_dataloader = self.dataloader("val", dataset)
        return self

    def get_train_dataset(self, data: Data):
        index = data.fold != self.fold
        if self.train_ratio < 1:
            index = get_subset(index, self.train_ratio)
        return self.get_dataset("train", data, index)

    def get_val_dataset(self, data: Data):
        index = data.fold == self.fold
        if self.val_ratio < 1:
            index = get_subset(index, self.val_ratio)
        return self.get_dataset("val", data, index)

    def get_dataset(self, mode: str, data, index):
        return self.dataset(
            mode, data.index[index], data.input[index], data.target[index]
        )


def get_subset(index, ratio):
    num_samples = len(index)
    num_subset = int(ratio * num_samples)
    idx = np.random.permutation(num_samples)[:num_subset]
    return index[idx]
