from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch.utils.data

import ivory.core.data


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


@dataclass
class DataLoader(ivory.core.data.DataLoader):
    dataset_class: Any = Dataset

    def create_train_val_dataloader(self):
        dataset = self.get_train_dataset()
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        dataset = self.get_val_dataset()
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
