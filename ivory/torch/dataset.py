import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input, target=None, index=None, transform=None):
        if index is not None:
            self.index = index
        else:
            self.index = np.arange(len(input))
        self.input = input
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = self.input[index]
        if self.transform:
            input = self.transform(input)
        if self.target is None:
            return self.index[index], input
        else:
            return self.index[index], input, self.target[index]

    @classmethod
    def from_dataframe(cls, df, input, target=None, transform=None):
        index = df.index.to_numpy()
        input = df[input].to_numpy()
        if target is not None:
            target = df[target].to_numpy()
        return cls(input, target, index, transform)
