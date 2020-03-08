from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor

from ivory.core.callback import Callback
from ivory.core.instance import get_attr
from ivory.torch.utils import cpu


@dataclass
class Metrics(Callback):
    criterion: Callable
    monitor: str = "val_loss"
    direction: str = "minimize"
    current_score: float = np.nan
    best_epoch: int = -1
    best_score: float = np.nan
    columns: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.criterion, str):
            self.criterion = get_attr(self.criterion)
        self.epoch_record = []

    def __str__(self):
        if len(self.epoch_record) == 0:
            return "None"
        record = self.epoch_record[-1]
        s = " ".join([f"{index}={record[index]:.04f}" for index in record.index])
        if record.name == self.best_epoch:
            s += " *"
        return s

    def on_epoch_start(self, obj):
        self.train_batch_record, self.val_batch_record = [], []

    def on_val_start(self, obj):
        self.index, self.output, self.target = [], [], []

    def train_step(self, index, output, target):
        loss = self.criterion(output, target)
        output = output.detach()
        self.train_batch_record.append(self.evaluate(loss.item(), output, target))
        return loss

    def val_step(self, index, output, target):
        loss = self.criterion(output, target)
        output = output.detach()
        self.val_batch_record.append(self.evaluate(loss.item(), output, target))
        if output.device.type != "cpu":
            output, target = cpu(output), cpu(target)
        self.index.append(index.numpy())
        self.output.append(output.numpy())
        self.target.append(target.numpy())

    def evaluate(self, loss: float, output: Tensor, target: Tensor) -> Dict[str, float]:
        return {"loss": loss}

    def on_epoch_end(self, obj):
        train_epoch_record = DataFrame(self.train_batch_record).mean(axis=0)
        val_epoch_record = DataFrame(self.val_batch_record).mean(axis=0)
        val_epoch_record.index = ["val_" + i for i in val_epoch_record.index]
        record = pd.concat([train_epoch_record, val_epoch_record])
        record.name = obj.trainer.epoch
        self.current_score = record[self.monitor]
        self.epoch_record.append(record)
        self.score = DataFrame(self.epoch_record)
        self.score.index.name = "epoch"
        if (
            self.best_score is np.nan
            or (self.direction == "minimize" and self.current_score < self.best_score)
            or (self.direction == "maximize" and self.current_score > self.best_score)
        ):
            self.best_score = self.current_score
            self.best_epoch = obj.trainer.epoch
            self.best_result = self.dataframe()
        self.log(obj)

    def log(self, obj):
        pass

    def dataframe(self):
        index = np.hstack(self.index)
        output = np.vstack(self.output)
        target = np.vstack(self.target)
        data = np.hstack([target, output])
        if self.columns is None:
            if output.shape[1] == 1:
                columns = ["true", "pred"]
            else:
                columns = [f"true{i}" for i in range(output.shape[1])]
                columns += [f"pred{i}" for i in range(output.shape[1])]
        else:
            columns = [f"{c}_true" for c in self.columns]
            columns + [f"{c}_pred" for c in self.columns]
        return DataFrame(data, index=index, columns=columns).sort_index()

    def state_dict(self):
        return {
            "score": self.score,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_result": self.best_result,
        }

    def load_state_dict(self, state_dict):
        self.score = state_dict['score']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.best_result = state_dict['best_result']
