from dataclasses import dataclass

import optuna
from termcolor import colored

from ivory.core.exceptions import EarlyStopped
from ivory.core.run import Run
from ivory.core.state import State
from ivory.utils.tqdm import tqdm


@dataclass
class Trainer(State):
    epoch: int = -1
    max_epochs: int = 1000
    global_step: int = -1
    verbose: int = 1

    def start(self, run: Run):
        """Starts a train or test loop.

        Args:
            run: a run instance.
        """
        if run.mode == "train":
            self.train(run)
        else:
            self.test(run)

    def train(self, run: Run):
        run.on_fit_start()
        try:
            self.loop(run)
        finally:
            run.on_fit_end()

    def test(self, run: Run):
        self.test_loop(run)

    def loop(self, run: Run):
        max_epoch = self.epoch + self.max_epochs
        epochs = range(self.epoch + 1, max_epoch + 1)
        if self.verbose == 1:
            epochs = tqdm(epochs, desc="Epoch", leave=False)
        early_stopped = pruned = None
        for self.epoch in epochs:
            if early_stopped or pruned:  # for tqdm
                continue
            run.on_epoch_start()
            self.train_loop(run)
            self.val_loop(run)
            try:
                run.on_epoch_end()
            except EarlyStopped as e:
                early_stopped = e
            except optuna.exceptions.TrialPruned as e:
                pruned = e
            finally:
                if self.verbose:
                    msg = self.message(run, max_epoch, early_stopped, pruned)
                    tqdm.write(msg)
        if pruned:
            raise pruned

    def get_dataloader(self, run: Run, mode: str):
        dataloader = run.dataloaders[mode]
        if self.verbose == 1:
            mode = "%-5s" % (mode[0].upper() + mode[1:])
            dataloader = tqdm(dataloader, desc=mode, leave=False)
        return dataloader

    def train_loop(self, run: Run):
        run.on_train_start()
        for index, input, target in self.get_dataloader(run, "train"):
            self.global_step += 1
            self.train_step(run, index, input, target)
        run.on_train_end()

    def val_loop(self, run: Run):
        run.on_val_start()
        for index, input, target in self.get_dataloader(run, "val"):
            self.val_step(run, index, input, target)
        run.on_val_end()

    def test_loop(self, run: Run):
        run.on_test_start()
        for index, input, *target in self.get_dataloader(run, "test"):
            self.test_step(run, index, input, *target)
        run.on_test_end()

    def train_step(self, run: Run, index, input, target):
        """Performs a single train step."""

    def val_step(self, run: Run, index, input, target):
        """Performs a single validation step."""

    def test_step(self, run: Run, index, input, *target):
        """Performs a single test step."""

    def message(self, run: Run, max_epoch: int, early_stopped, pruned) -> str:
        width = len(str(max_epoch))
        epoch = str(self.epoch).zfill(width)
        msg = f"[{run.name}] epoch={epoch} {run.metrics}"
        if run.monitor.is_best:
            msg = colored(msg, "green")
        else:
            msg = colored(msg, "yellow")
        if early_stopped:
            msg += colored(" early stopped", "magenta")
        if pruned:
            msg += colored(" pruned", "red")
        return msg
