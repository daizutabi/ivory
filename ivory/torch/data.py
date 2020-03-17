from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader


@dataclass
class Dataset(torch.utils.data.Dataset):
    mode: str
    index: np.ndarray
    input: Any
    target: Any = None
    transform: Optional[Callable[[str, Any, Any], Tuple[Any, Any]]] = None

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mode={self.mode}, num_samples={len(self)})"

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        input = self.input[index]
        if self.target is not None:
            target = self.target[index]
        else:
            target = None
        if self.transform:
            input, target = self.transform(self.mode, input, target)
        if target is None:
            return self.index[index], input
        else:
            return self.index[index], input, target


def dataloader(mode: str, dataset, batch_size=32):
    if mode == "train":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
