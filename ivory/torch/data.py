from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np
import torch.utils.data
from pandas import DataFrame
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input, target=None, index=None, transform=None):
        if index is not None:
            self.index = index
        else:
            self.index = np.arange(len(input))
        self.input = input
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = self.input[index]
        if self.transform:
            input = self.transform(input)
        if self.target is None:
            return self.index[index], input
        else:
            return self.index[index], input, self.target[index]

    @classmethod
    def from_dataframe(cls, df, input, target=None, transform=None):
        index = df.index.to_numpy()
        input = df[input].to_numpy()
        if target is not None:
            target = df[target].to_numpy()
        return cls(input, target, index, transform)


@dataclass
class DataLoaders:
    data: Any
    transform: Optional[Callable] = None
    batch_size: int = 32

    def get_train_dataset(self, fold: int):
        raise NotImplementedError

    def get_val_dataset(self, fold: int):
        raise NotImplementedError

    def __getitem__(self, fold: int):
        dataset = self.get_train_dataset(fold)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataset = self.get_val_dataset(fold)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader


@dataclass
class DataFrameLoaders(DataLoaders):
    data: DataFrame
    input: List[str] = field(default_factory=list)
    target: List[str] = field(default_factory=list)

    def get_train_dataset(self, fold: int):
        df = self.data.query("fold != @fold")
        return Dataset.from_dataframe(df, self.input, self.target, self.transform)

    def get_val_dataset(self, fold: int):
        df = self.data.query("fold == @fold")
        return Dataset.from_dataframe(df, self.input, self.target)
