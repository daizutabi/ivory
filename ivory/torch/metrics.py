from dataclasses import dataclass, field
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
    direction: str = "minimize"  # minimize or maximize
    current_score: float = np.nan
    best_epoch: int = -1
    best_score: float = np.nan
    best_output: Optional[DataFrame] = field(default=None, init=False, repr=False)
    history: Optional[DataFrame] = field(default=None, init=False, repr=False)
    columns: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.criterion, str):
            self.criterion = get_attr(self.criterion)
        self.epoch_record = []

    @property
    def latest(self):
        record = self.epoch_record[-1]
        s = " ".join([f"{index}={record[index]:.04f}" for index in record.index])
        if self.current_score == self.best_score:
            s += " *"
        return s

    def on_epoch_start(self, run):
        self.train_batch_record, self.val_batch_record = [], []

    def on_val_start(self, run):
        self.batch_index, self.batch_output = [], []

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
            output = cpu(output)
        self.batch_index.append(index.numpy())
        self.batch_output.append(output.numpy())

    def evaluate(self, loss: float, output: Tensor, target: Tensor) -> Dict[str, float]:
        return {"loss": loss}

    def on_epoch_end(self, run):
        train_epoch_record = DataFrame(self.train_batch_record).mean(axis=0)
        val_epoch_record = DataFrame(self.val_batch_record).mean(axis=0)
        val_epoch_record.index = ["val_" + i for i in val_epoch_record.index]
        record = pd.concat([train_epoch_record, val_epoch_record])
        record.name = run.trainer.epoch
        self.current_score = record[self.monitor]
        self.epoch_record.append(record)
        self.history = DataFrame(self.epoch_record)
        self.history.index.name = "epoch"
        if (
            self.best_score is np.nan
            or (self.direction == "minimize" and self.current_score < self.best_score)
            or (self.direction == "maximize" and self.current_score > self.best_score)
        ):
            self.best_score = self.current_score
            self.best_epoch = run.trainer.epoch
            self.best_output = self.output
        self.log(run)

    def log(self, run):
        pass

    @property
    def output(self):
        index, output = np.hstack(self.batch_index), np.vstack(self.batch_output)
        if self.columns is None:
            if output.shape[1] == 1:
                columns = ["output"]
            else:
                columns += [f"output.{i}" for i in range(output.shape[1])]
        return DataFrame(output, index=index, columns=columns).sort_index()

    def state_dict(self):
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_output": self.best_output,
            "history": self.history,
        }

    def load_state_dict(self, state_dict):
        self.best_score = state_dict["best_score"]
        self.best_epoch = state_dict["best_epoch"]
        self.best_output = state_dict["best_output"]
        self.history = state_dict["history"]
