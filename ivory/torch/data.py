from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, List, Optional

import numpy as np
import torch.utils.data
from pandas import DataFrame
from torch.utils.data import DataLoader, random_split


@dataclass
class Dataset(torch.utils.data.Dataset):
    input: Any
    target: Any = None
    index: np.ndarray = None
    transform: Optional[Callable] = None

    def __post_init__(self):
        if self.index is None:
            self.index = np.arange(len(self.input))

    def __repr__(self):
        cls_name = self.__class__.__name__
        s = f"{cls_name}(num_samples={len(self)}, input_shape={self.input.shape[1:]}, "
        if self.target is None:
            s += f"target_shape=None, transform={self.transform})"
        else:
            s += f"target_shape={self.target.shape[1:]}, transform={self.transform})"
        return s

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = self.input[index]
        target = self.target[index] if self.target is not None else None
        if self.transform:
            input, target = self.transform(input, target)
        if target is None:
            return self.index[index], input
        else:
            return self.index[index], input, target

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
    batch_size: int
    transform: Optional[Callable] = None
    train_percent_check: float = 1.0
    val_percent_check: float = 1.0

    def get_train_dataset(self, fold: int):
        raise NotImplementedError

    def get_val_dataset(self, fold: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, fold: int):
        dataset = self.get_train_dataset(fold)
        if self.train_percent_check < 1.0:
            partial = int(self.train_percent_check * len(dataset))
            dataset, _ = random_split(dataset, [partial, len(dataset) - partial])
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataset = self.get_val_dataset(fold)
        if self.val_percent_check < 1.0:
            partial = int(self.val_percent_check * len(dataset))
            dataset, _ = random_split(dataset, [partial, len(dataset) - partial])
        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader


@dataclass
class DataFrameLoaders(DataLoaders):
    data: DataFrame
    input: List[str] = field(default_factory=list)
    target: List[str] = field(default_factory=list)

    def __repr__(self):
        cls_name = self.__class__.__name__
        s = f"{cls_name}(data_shape={self.data.shape}, input={self.input}, "
        s += f"target={self.target}, transform={self.transform})"
        return s

    def __len__(self):
        return self.data.fold.max() + 1

    def get_train_dataset(self, fold: int):
        df = self.data.query("fold != @fold")
        transform = self.transform and partial(self.transform, mode="train")
        return Dataset.from_dataframe(df, self.input, self.target, transform)

    def get_val_dataset(self, fold: int):
        df = self.data.query("fold == @fold")
        transform = self.transform and partial(self.transform, mode="val")
        return Dataset.from_dataframe(df, self.input, self.target, transform)
