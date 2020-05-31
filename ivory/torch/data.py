from dataclasses import dataclass

import torch.utils.data
from torch.utils.data import DataLoader

import ivory.core.data


@dataclass(repr=False)
class Dataset(ivory.core.data.Dataset, torch.utils.data.Dataset):
    pass


class DataLoaders(ivory.core.data.DataLoaders):
    def get_dataloader(self, dataset, batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
