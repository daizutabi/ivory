from dataclasses import dataclass

import optuna
from termcolor import colored
from tqdm import tqdm

from ivory.core.exceptions import EarlyStopped
from ivory.core.state import State


@dataclass
class Trainer(State):
    epoch: int = -1
    max_epochs: int = 1000
    global_step: int = -1
    verbose: int = 1

    def train_loop(self, run):
        run.on_train_start()
        dataloader = run.dataloaders.train
        if self.verbose == 1:
            dataloader = tqdm(dataloader, desc="Train", leave=False)
        for index, input, target in dataloader:
            self.global_step += 1
            self.train_step(index, input, target, run)
        run.on_train_end()

    def train_step(self, index, input, target, run):
        pass

    def val_loop(self, run):
        run.on_val_start()
        dataloader = run.dataloaders.val
        if self.verbose == 1:
            dataloader = tqdm(dataloader, desc="Val  ", leave=False)
        for index, input, target in dataloader:
            self.val_step(index, input, target, run)
        run.on_val_end()

    def val_step(self, index, input, target, run):
        pass

    def loop(self, run, leave):
        max_epoch = self.epoch + self.max_epochs
        epochs = range(self.epoch + 1, max_epoch + 1)
        if self.verbose == 1:
            epochs = tqdm(epochs, desc="Epoch", leave=leave)
        early_stopped = None
        pruned = None
        for self.epoch in epochs:
            if early_stopped or pruned:  # for tqdm
                continue
            run.on_epoch_start()
            self.train_loop(run)
            if run.dataloaders.val:
                self.val_loop(run)
            try:
                run.on_epoch_end()
            except EarlyStopped as e:
                early_stopped = e
            except optuna.exceptions.TrialPruned as e:
                pruned = e
            finally:
                if self.verbose:
                    msg = self.message(max_epoch, early_stopped, pruned, run=run)
                    tqdm.write(msg)
        if pruned:
            raise pruned

    def fit(self, run, leave=True):
        run.on_fit_start()
        try:
            self.loop(run, leave)
        finally:
            run.on_fit_end()

    def test(self, run, leave=False):
        run.on_test_start()
        dataloader = run.dataloaders.test
        if self.verbose == 1:
            dataloader = tqdm(dataloader, desc="Test ", leave=leave)
        for index, input in dataloader:
            self.test_step(index, input, run)
        run.on_test_end()

    def test_step(self, index, input, run):
        pass

    def message(self, max_epoch, early_stopped, pruned, run):
        width = len(str(max_epoch))
        epoch = str(self.epoch).zfill(width)
        msg = f"[{run.name}] epoch={epoch} {run.metrics}"
        if run.monitor.is_best:
            msg = colored(msg, "green")
        else:
            msg = colored(msg, "yellow")
        if early_stopped:
            msg += colored(" early stopped", "red")
        if pruned:
            msg += colored(" pruned", "red")
        return msg
