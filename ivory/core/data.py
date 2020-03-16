from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from pandas import DataFrame

from ivory.core import instance


@dataclass
class Data:
    load: Callable[[], Tuple[Any, Any]]
    fold_name: str = "fold"

    def __post_init__(self):
        self.index = None
        self.input = None
        self.target = None
        self.fold = None

    def __repr__(self):
        cls_name = self.__class__.__name__
        if self.index is None:
            return f"{cls_name}(<data not created>)"
        s = f"{cls_name}(num_folds={self.fold.max()+1}, num_samples={len(self.index)})"
        return s

    def to_numpy(self, df: DataFrame):
        self.index = df.index.to_numpy()
        cs = df.columns
        if self.fold_name in cs:
            self.fold = df[self.fold_name].to_numpy()
            cs = [c for c in cs if c != self.fold_name]
        return df[cs].to_numpy()

    def create_data(self):
        self.input, self.target = self.load()
        if isinstance(self.input, DataFrame):
            self.input = self.to_numpy(self.input)
        if isinstance(self.target, DataFrame):
            self.target = self.to_numpy(self.target)
        if self.index is None:
            raise NotImplementedError


@dataclass
class DataLoader:
    data: Data
    fold: int = 0
    transform: Optional[Callable[[str, Any, Any], Tuple[Any, Any]]] = None
    batch_size: int = 32
    train_ratio: float = 1.0
    val_ratio: float = 1.0
    dataset_class: Any = None
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.data.index is None:
            self.data.create_data()
        if isinstance(self.dataset_class, str):
            self.dataset_class = instance.get_attr(self.dataset_class)
        self.create_train_val_dataloader()

    def __repr__(self):
        cls_name = self.__class__.__name__
        s = f"{cls_name}(data={self.data}, fold={self.fold}"
        if self.transform:
            s += f", transform={self.transform}"
        return s + f", batch_size={self.batch_size}"

    def create_train_val_dataloader(self):
        self.train_dataloader = None
        self.val_dataloader = None

    def get_train_dataset(self):
        index = self.data.fold != self.fold
        if self.train_ratio < 1:
            index = get_subset(index, self.train_ratio)
        return self.get_dataset("train", index)

    def get_val_dataset(self):
        if self.val_ratio == 0:
            return None
        index = self.data.fold == self.fold
        if self.val_ratio < 1:
            index = get_subset(index, self.val_ratio)
        return self.get_dataset("val", index)

    def get_dataset(self, mode: str, index):
        return self.dataset_class(
            mode,
            self.data.index[index],
            self.data.input[index],
            self.data.target[index],
            self.transform,
            **self.dataset_kwargs,
        )


def get_subset(index, ratio):
    num_samples = len(index)
    num_subset = int(ratio * num_samples)
    idx = np.random.permutation(num_samples)[:num_subset]
    return index[idx]
