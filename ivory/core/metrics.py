from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from ivory.core.callbacks import Callback
from ivory.core.instance import get_attr


@dataclass
class Metrics(Callback):
    criterion: Callable
    monitor: str = "val_loss"
    mode: str = "min"  # min or max
    current_epoch: int = -1
    current_record: Optional[Series] = field(default=None, init=False, repr=False)
    current_score: float = np.nan
    current_output: Optional[DataFrame] = field(default=None, init=False, repr=False)
    best_epoch: int = -1
    best_record: Optional[Series] = field(default=None, init=False, repr=False)
    best_score: float = np.nan
    best_output: Optional[DataFrame] = field(default=None, init=False, repr=False)
    history: Optional[DataFrame] = field(default=None, init=False, repr=False)
    columns: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.criterion, str):
            self.criterion = get_attr(self.criterion)
        self.best_score = np.inf if self.mode == "min" else -np.inf

    @property
    def latest(self):
        record = self.history.iloc[-1]
        s = " ".join([f"{index}={record[index]:.04f}" for index in record.index])
        if self.current_score == self.best_score:
            s += " *"
        return s

    def on_epoch_start(self, run):
        self.train_batch_record, self.val_batch_record = [], []

    def on_val_start(self, run):
        self.batch_index, self.batch_output = [], []

    def train_step(self, index, output, target):
        loss, record = self.train_evaluate(output, target)
        self.train_batch_record.append(record)
        return loss

    def val_step(self, index, output, target):
        output, record = self.val_evaluate(output, target)
        self.val_batch_record.append(record)
        self.batch_index.append(index.numpy())
        self.batch_output.append(output.numpy())

    def on_epoch_end(self, run):
        train_epoch_record = DataFrame(self.train_batch_record).mean(axis=0)
        val_epoch_record = DataFrame(self.val_batch_record).mean(axis=0)
        val_epoch_record.index = ["val_" + i for i in val_epoch_record.index]
        self.current_epoch = run.trainer.epoch
        self.current_record = pd.concat([train_epoch_record, val_epoch_record])
        self.current_record.name = self.current_epoch
        self.current_score = self.current_record[self.monitor]
        self.current_output = self.output
        if self.history is None:
            self.history = self.current_record.to_frame().T
            self.history.index.name = "epoch"
        else:
            self.history = self.history.append(self.current_record)
        if self.is_best:
            self.best_score = self.current_score
            self.best_epoch = run.trainer.epoch
            self.best_output = self.current_output
        self.log(run)

    def log(self, run):
        pass

    @property
    def is_best(self):
        if self.mode == "min":
            return self.current_score < self.best_score
        else:
            return self.current_score > self.best_score

    @property
    def output(self):
        index, output = np.hstack(self.batch_index), np.vstack(self.batch_output)
        if self.columns is None:
            columns = [f"output.{i}" for i in range(output.shape[1])]
            if len(columns) == 1:
                columns = ["output"]
        else:
            columns = self.columns
        return DataFrame(output, index=index, columns=columns).sort_index()

    def state_dict(self):
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_output": self.best_output,
            "current_output": self.current_output,
            "history": self.history,
        }

    def load_state_dict(self, state_dict):
        self.best_score = state_dict["best_score"]
        self.best_epoch = state_dict["best_epoch"]
        self.best_output = state_dict["best_output"]
        self.current_output = state_dict["current_output"]
        self.history = state_dict["history"]
