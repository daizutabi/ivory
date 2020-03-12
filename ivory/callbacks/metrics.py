from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from ivory.callbacks import Callback
from ivory.core import instance
from ivory.core.state import State


@dataclass
class Metrics(Callback, State):
    criterion: Callable
    monitor: str = "val_loss"
    mode: str = "min"

    def __post_init__(self):
        if isinstance(self.criterion, str):
            self.criterion = instance.get_attr(self.criterion)
        self.best_epoch = -1
        if self.mode == "min":
            self.best_score = np.inf
        else:
            self.best_score = -np.inf
        self.best_output = None
        self.history = None

    def on_epoch_start(self, run):
        self.train_batch_record, self.val_batch_record = [], []

    def on_val_start(self, run):
        self.val_batch_index, self.val_batch_output = [], []

    def train_step(self, index, output, target):
        loss, record = self.train_evaluate(index, output, target)
        self.train_batch_record.append(record)
        return loss

    def train_evaluate(self, index, output, target):
        """Returns a tuple of (loss, record)."""
        raise NotImplementedError

    def val_step(self, index, output, target):
        index, output, record = self.val_evaluate(index, output, target)
        self.val_batch_index.append(index)
        self.val_batch_output.append(output)
        self.val_batch_record.append(record)

    def val_evaluate(self, index, output, target):
        """Returns a tuple of (index, output, record)."""
        raise NotImplementedError

    def on_epoch_end(self, run):
        train_epoch_record = DataFrame(self.train_batch_record).mean(axis=0)
        val_epoch_record = DataFrame(self.val_batch_record).mean(axis=0)
        val_epoch_record.index = ["val_" + i for i in val_epoch_record.index]
        self.current_epoch = run.trainer.epoch
        self.current_record = pd.concat([train_epoch_record, val_epoch_record])
        self.current_record.name = self.current_epoch
        self.on_current_record(run)
        self.current_score = self.current_record[self.monitor]
        if self.history is None:
            self.history = self.current_record.to_frame().T
            self.history.index.name = "epoch"
        else:
            self.history = self.history.append(self.current_record)
        if self.is_best:
            self.best_epoch = self.current_epoch
            self.best_score = self.current_score
            self.best_output = self.output
        self.log(run)

    def on_current_record(self, run):
        pass

    def log(self, run):
        pass

    @property
    def is_best(self):
        if self.mode == "min":
            return self.current_score <= self.best_score  # = is needed for tracking.
        else:
            return self.current_score >= self.best_score

    @property
    def latest(self):
        record = self.history.iloc[-1]
        s = " ".join([f"{index}={record[index]:.2e}" for index in record.index])
        if self.current_score == self.best_score:
            s += " *"
        return s

    @property
    def output(self):
        index = np.hstack(self.val_batch_index)
        output = np.vstack(self.val_batch_output)
        columns = [f"output.{i}" for i in range(output.shape[1])]
        if len(columns) == 1:
            columns = ["output"]
        return DataFrame(output, index=index, columns=columns).sort_index()
