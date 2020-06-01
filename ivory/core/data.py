"""
Ivory uses three classes for data presentation: `Data`, `Dataset`, and `Datasets`.

Basically, you only need to define a class that is a subclass of `Data`
and use original `Dataset` and `Datasets`. An example parameter YAML file is:

    datasets:
      data:
        class: your.Data  # a subclass of ivory.core.data.Data
      dataset:
      fold: 0

But if you need, you can define your `Dataset` and/or `Datasets`.

    datasets:
      class: your.Datasets
      data:
        class: your.Data  # a subclass of ivory.core.data.Data
      dataset:
        def: your.Dataset
      fold: 0

Note:
    Use a `'def'` key for `dataset` instead of `'class'`.
    See [Tutorial](/tutorial/data)
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

import ivory.core.collections
from ivory.core import instance


@dataclass
class Data:
    """Base class to provide data to a `Dataset` instance.

    To make a subclass, you need to assign the following attributes in
    the `Data.init()` method:

    * `index`: Index of samples.
    * `input`: Input data.
    * `target`: Target data.
    * `fold`: Fold number.
    """

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
        """Initializes `index`, `input`, `target`, and `fold` attributes.

        The fold number of test data must be `-1`.

        Examples:
            For regression

                def init(self):
                    self.index = np.range(100)
                    self.input = np.random.randn(100, 5)
                    self.target = np.random.randn(100)
                    self.fold = np.random.randint(5)
                    self.fold[80:] = -1

            For classification

                def init(self):
                    self.index = np.range(100)
                    self.input = np.random.randn(100, 5)
                    self.target = np.random.randint(100, 10)
                    self.fold = np.random.randint(5)
                    self.fold[80:] = -1
        """

    def get_index(self, mode: str, fold: int) -> np.ndarray:
        """Returns index according to the mode and fold.

        Args:
            mode: Mode name: `'train'`, `'val'`, or `'test'`.
            fold: Fold number.
        """
        index = np.arange(len(self.fold))
        if mode == "train":
            return index[(self.fold != fold) & (self.fold != -1)]
        elif mode == "val":
            return index[self.fold == fold]
        else:
            return index[self.fold == -1]

    def get_input(self, index):
        """Returns input data.

        By default, this method returns `self.input[index]`. You can override this
        behavior in a subclass.

        Args:
            index (int or 1D-array): Index.
        """
        return self.input[index]

    def get_target(self, index):
        """Returns target data.

        By default, this method returns `self.target[index]`. You can override this
        behavior in a subclass.

        Args:
            index (int or 1D-array): Index.
        """
        return self.target[index]

    def get(self, index) -> Tuple:
        """Returns a tuple of (`index`, `input`, `target`) according to the index.

        Args:
            index (int or 1D-array): Index.
        """
        return self.index[index], self.get_input(index), self.get_target(index)


@dataclass
class Dataset:
    """Dataset class represents a set of data for a mode and fold.

    Args:
        data: `Data` instance that provides data to `Dataset` instance.
        mode: Mode name: `'train'`, `'val'`, or `'test'`.
        fold: Fold number.
        transform (callable, optional): Callable to transform the data.

    The `transform` must take 2 or 3 arguments: (`mode`, `input`, optional
    `target`) and return a tuple of (`input`, optional `target`).
    """

    data: Data
    mode: str
    fold: int
    transform: Optional[Callable] = None

    def __post_init__(self):
        self.index = self.data.get_index(self.mode, self.fold)
        if self.mode == "test":
            self.fold = -1
        if self.transform:
            self.transform = instance.get_attr(self.transform)
        self.init()

    def init(self):
        """Called at initialization. You can add any process in a subclass."""
        pass

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mode={self.mode!r}, num_samples={len(self)})"

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        if index == slice(None, None, None):
            index, input, *target = self.get()
        else:
            index, input, *target = self.get(index)
        if self.transform:
            input, *target = self.transform(self.mode, input, *target)
        return (index, input, *target)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def get(self, index=None) -> Tuple:
        """Returns a tuple of (`index`, `input`, `target`) according to the index.

        If index is `None`, reutrns all of the data.

        Args:
            index (int or 1D-array, optional): Index.
        """
        if index is None:
            return self.data.get(self.index)
        else:
            return self.data.get(self.index[index])

    def sample(self, n: int = 0, frac: float = 0.0) -> Tuple:
        """Returns a tuple of (`index`, `input`, `target`) randomly sampled.

        Args:
            n: Size of sampling.
            frac: Ratio of sampling.
        """
        index, input, *target = self[:]
        if frac:
            n = int(len(index) * frac)
        idx = np.random.permutation(len(index))[:n]
        return tuple([x[idx] for x in [index, input, *target]])


@dataclass
class Datasets(ivory.core.collections.Dict):
    """Dataset class represents a collection of `Dataset` for a fold.

    Args:
        data: `Data` instance that provides data to `Dataset` instance.
        dataset: Dataset factory.
        fold: Fold number.

    Attributes:
        train (Dataset): Train dataset.
        val (Dataset): Validation dataset.
        test (Dataset): Test dataset.
        fold: Fold number.
    """

    data: Data
    dataset: Callable
    fold: int

    def __post_init__(self):
        super().__init__()
        for mode in ["train", "val", "test"]:
            self[mode] = self.dataset(self.data, mode, self.fold)


class DataLoaders(ivory.core.collections.Dict):
    """DataLoaders class represents a collection of `DataLoader`.

    Args:
        datasets: `Datasets` instance.
        batch_size: Batch_size
        shuffle: If True, train dataset is shuffled.

    Attributes:
        train (Dataset): Train dataset.
        val (Dataset): Validation dataset.
        test (Dataset): Test dataset.
    """

    def __init__(self, datasets: Datasets, batch_size: int, shuffle: bool):
        super().__init__()
        for mode in ["train", "val", "test"]:
            self[mode] = self.get_dataloader(datasets[mode], batch_size, shuffle)
            shuffle = False

    def get_dataloader(self, dataset, batch_size, shuffle):
        raise NotImplementedError
