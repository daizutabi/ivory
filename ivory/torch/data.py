from dataclasses import dataclass

import torch.utils.data

import ivory.core.data


@dataclass(repr=False)
class Dataset(ivory.core.data.Dataset, torch.utils.data.Dataset):
    pass


@dataclass(repr=False)
class DataLoaders(ivory.core.data.DataLoaders):
    def get_dataloader(self, dataset, mode):
        shuffle = True if mode == "train" else False
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )
