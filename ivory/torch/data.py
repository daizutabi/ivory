from dataclasses import dataclass

import torch.utils.data
from torch.utils.data import DataLoader

import ivory.core.data


@dataclass(repr=False)
class Dataset(ivory.core.data.Dataset, torch.utils.data.Dataset):
    pass


@dataclass(repr=False)
class DataLoaders(ivory.core.data.DataLoaders):
    num_workers: int = 0
    pin_memory: bool = False

    def get_dataloader(self, dataset, batch_size, shuffle):
        drop_last = dataset.mode == "train"
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
