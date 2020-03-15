from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import random_split

from ivory.core import instance


@dataclass
class Dataset(torch.utils.data.Dataset):
    input: Any
    target: Any = None
    transform: Optional[Callable] = None
    index: np.ndarray = None
    mode: str = "train"

    def __post_init__(self):
        if isinstance(self.input, DataFrame):
            self.index = self.input.index.to_numpy()
            self.input = self.input.to_numpy()
        if isinstance(self.target, DataFrame):
            self.index = self.target.index.to_numpy()
            self.target = self.target.to_numpy()
        if self.index is None:
            self.index = np.arange(len(self.input))

    def __repr__(self):
        cls_name = self.__class__.__name__
        s = f"{cls_name}(num_samples={len(self)}, input_shape={self.input.shape[1:]}"
        if self.target is not None:
            s += f", target_shape={self.target.shape[1:]}"
        if self.transform is not None:
            s += f", transform={self.transform}"
        return s + f", mode='{self.mode}')"

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = self.input[index]
        target = self.target[index] if self.target is not None else None
        if self.transform:
            input, target = self.transform(input, target, self.mode)
        if target is None:
            return self.index[index], input
        else:
            return self.index[index], input, target


@dataclass
class DataLoader:
    load: Callable
    transform: Optional[Callable] = None
    batch_size: int = 32
    train_percent_check: float = 1.0
    val_percent_check: float = 1.0
    dataset_class: type = Dataset
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    fold_name: str = "fold"

    def __post_init__(self):
        self.index = None
        self.fold = None
        self.input = None
        self.target = None
        if isinstance(self.dataset_class, str):
            self.dataset_class = instance.get_attr(self.dataset_class)

    def load_data(self):
        def to_numpy(df):
            self.index = df.index.to_numpy()
            cs = df.columns
            if self.fold_name in cs:
                self.fold = df[self.fold_name].to_numpy()
                cs = [c for c in df if c != self.fold_name]
            return df[cs].to_numpy()

        input, target = self.load()
        if isinstance(input, DataFrame):
            self.input = to_numpy(input)
        if isinstance(target, DataFrame):
            self.target = to_numpy(target)

    def __repr__(self):
        cls_name = self.__class__.__name__
        if self.index is None:
            return f"{cls_name}(<data not loaded>)"
        s = f"{cls_name}(num_folds={len(self)}, num_samples={len(self.input)}"
        s += f", input_shape={self.input.shape[1:]}"
        if self.target is not None:
            s += f", target_shape={self.target.shape[1:]}"
        return s + ")"

    def __len__(self):
        if self.index is None:
            return 0
        return self.fold.max() + 1

    def __getitem__(self, fold: int):
        if self.index is None:
            self.load_data()
        dataset = self.get_train_dataset(fold)
        if self.train_percent_check < 1.0:
            subsize = int(self.train_percent_check * len(dataset))
            dataset, _ = random_split(dataset, [subsize, len(dataset) - subsize])
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        dataset = self.get_val_dataset(fold)
        if self.val_percent_check < 1.0:
            subsize = int(self.val_percent_check * len(dataset))
            dataset, _ = random_split(dataset, [subsize, len(dataset) - subsize])
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def get_train_dataset(self, fold: int):
        idx = self.fold != fold
        return self.dataset_class(
            self.input[idx],
            self.target[idx],
            self.transform,
            self.index[idx],
            **self.dataset_kwargs,
            mode="train",
        )

    def get_val_dataset(self, fold: int):
        idx = self.fold == fold
        return self.dataset_class(
            self.input[idx],
            self.target[idx],
            self.transform,
            self.index[idx],
            **self.dataset_kwargs,
            mode="val",
        )
