from dataclasses import dataclass

from nnabla.utils.data_iterator import data_iterator_simple

import ivory.core.data
from ivory.core.data import Dataset


@dataclass
class DataLoader:
    dataset: Dataset
    batch_size: int
    shuffle: bool = False
    with_memory_cache: bool = False
    with_file_cache: bool = False

    def __post_init__(self):
        self.iterator = data_iterator_simple(
            self.load_func,
            len(self.dataset),
            self.batch_size,
            shuffle=self.shuffle,
            with_memory_cache=self.with_memory_cache,
            with_file_cache=self.with_file_cache,
        )

    def __len__(self):
        if len(self.dataset) % self.batch_size:  # FIXME
            raise NotImplementedError
        return len(self.dataset) // self.batch_size

    def load_func(self, index):
        return self.dataset[index]

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class DataLoaders(ivory.core.data.DataLoaders):
    def get_dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
