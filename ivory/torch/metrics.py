import hydra
import numpy as np
from pandas import DataFrame


class Metrics:
    def __init__(self, criterion):
        if isinstance(criterion, str):
            criterion = hydra.utils.get_method(criterion)
        self.criterion = criterion

    def reset(self):
        self.index, self.output, self.target, self.loss = [], [], [], []

    def step(self, index, output, target):
        loss = self.criterion(output, target)
        if output.device.type != "cpu":
            index, output = index.cpu(), output.cpu()
        self.index.append(index.numpy())
        self.output.append(output.detach().numpy())
        self.target.append(target.numpy())
        self.loss.append(loss.item())
        return loss

    def dataframe(self, columns=None):
        index = np.hstack(self.index)
        output = np.vstack(self.output)
        target = np.vstack(self.target)
        data = np.hstack([target, output])
        if columns is None:
            if output.shape[1] == 1:
                columns = ["true", "pred"]
            else:
                columns = [f"true{i}" for i in range(output.shape[1])]
                columns += [f"pred{i}" for i in range(output.shape[1])]
        else:
            columns = [f"{c}_true" for c in columns] + [f"{c}_pred" for c in columns]
        return DataFrame(data, index=index, columns=columns).sort_index()
