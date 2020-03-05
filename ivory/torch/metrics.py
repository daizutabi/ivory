import hydra
import numpy as np
from pandas import DataFrame


class Metrics:
    def __init__(self, criterion):
        if isinstance(criterion, str):
            criterion = hydra.utils.get_method(criterion)
        self.criterion = criterion

    def reset(self):
        self.index, self.output, self.loss = [], [], []

    def step(self, index, output, target):
        loss = self.criterion(output, target)
        if output.device.type != "cpu":
            index, output = index.cpu(), output.cpu()
        self.index.append(index.numpy())
        self.output.append(output.detach().numpy())
        self.loss.append(loss.item())
        return loss

    def dataframe(self, columns=None):
        index = np.hstack(self.index)
        output = np.vstack(self.output)
        if columns is None:
            if output.shape[1] == 1:
                columns = ["pred"]
            else:
                columns = [f"pred{i}" for i in range(output.shape[1])]
        return DataFrame(output, index=index, columns=columns).sort_index()
