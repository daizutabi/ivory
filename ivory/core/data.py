import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


@dataclass
class Data:
    mode: str = "train"
    initialized: bool = False

    def __post_init__(self):
        self.fold = None

    def initialize(self):
        self.init()
        self.initialized = True

    def init(self):
        """Initialzes the data. For example, read a csv file as a DataFrame.

        Called from ivory.core.data.Data.
        """
        raise NotImplementedError

    def get(self, index=None):
        """Returns a subset of data according to `mode` and `index`.

        Returned object can be any type but should be processed by Dataset's ``get()``.

        Args:
            index (list): 1d-array of bool, optional. The length is the same as `fold`.

        Called from ivory.core.data.DataLoader.
        """
        raise NotImplementedError


@dataclass
class Dataset:
    mode: str
    data: Any
    transform: Optional[Callable[[str, Any, Any], Tuple[Any, Any]]] = None

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mode={self.mode}, num_samples={len(self)})"

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        index, input, *target = self.get(index)
        if self.transform:
            input, *target = self.transform(self.mode, input, *target)
        return [index, input, *target]

    def get(self, index):
        """Returns a tuple of (index, input, target) or (index, input)."""
        raise NotImplementedError


@dataclass
class DataLoader:
    dataset: Callable
    fold: int = 0
    batch_size: int = 32

    def __post_init__(self):
        self._dataloaders = {"train": None, "val": None, "test": None}

    def __repr__(self):
        cls_name = self.__class__.__name__
        if isinstance(self.dataset, functools.partial):
            dataset = self.dataset.func.__module__
            dataset += "." + self.dataset.func.__name__
            kwargs = [f"{key}={value}" for key, value in self.dataset.keywords.items()]
            kwargs = ", ".join(kwargs)
        else:
            dataset = self.dataset.__module__
            dataset += "." + self.dataset.__name__
            kwargs = ''
        s = f"{cls_name}(dataset={dataset}({kwargs}), fold={self.fold}, "
        return s + f"batch_size={self.batch_size})"

    def init(self, data: Data):
        if not data.initialized:
            data.initialize()
        if data.mode == "train":
            for mode in ["train", "val"]:
                index = self.get_index(mode, data)
                dataset = self.dataset(mode, data.get(index))
                self._dataloaders[mode] = self.get_dataloader(mode, dataset)
        else:
            dataset = self.dataset("test", data.get())
            self._dataloaders["test"] = self.get_dataloader("test", dataset)

    def get_index(self, mode, data):
        if mode == "train":
            return data.fold != self.fold
        else:
            return data.fold == self.fold

    def get_dataloader(self, mode, dataset):
        return dataset

    @property
    def train(self):
        return self._dataloaders["train"]

    @property
    def val(self):
        return self._dataloaders["val"]

    @property
    def test(self):
        return self._dataloaders["test"]
