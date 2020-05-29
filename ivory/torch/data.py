from dataclasses import dataclass

import torch.utils.data

import ivory.core.data


@dataclass(repr=False)
class Dataset(ivory.core.data.Dataset, torch.utils.data.Dataset):
    pass
