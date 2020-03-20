from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch.utils.data

import ivory.core.data


@dataclass
class Dataset(torch.utils.data.Dataset):
    mode: str
    data: Any
    transform: Optional[Callable[[str, Any, Any], Tuple[Any, Any]]] = None

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mode={self.mode}, num_samples={len(self)})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index, input, target = self.get(index)
        if self.transform:
            input, target = self.transform(self.mode, input, target)
        if target is None:
            return index, input
        else:
            return index, input, target

    def get(self, index) -> Tuple[Any, Any, Any]:
        """Returns a tuple of (data index, input, target)."""
        raise NotImplementedError


@dataclass
class DataLoader(ivory.core.data.DataLoader):
    def get_train_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def get_val_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )
