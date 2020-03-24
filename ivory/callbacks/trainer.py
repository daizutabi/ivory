from dataclasses import dataclass

from tqdm import tqdm

from ivory.core.state import State


@dataclass
class Trainer(State):
    epoch: int = -1
    max_epochs: int = 1000
    global_step: int = -1
    verbose: int = 1

    def train_step(self, index, input, target, run):
        pass

    def val_step(self, index, input, target, run):
        pass

    def train_loop(self, run):
        run.on_train_start()
        dataloader = run.dataloader.train
        if self.verbose == 1:
            dataloader = tqdm(dataloader, desc="-Train  ", leave=False)
        for index, input, target in dataloader:
            self.global_step += 1
            self.train_step(index, input, target, run)
        run.on_train_end()

    def val_loop(self, run):
        run.on_val_start()
        dataloader = run.dataloader.val
        if self.verbose == 1:
            dataloader = tqdm(dataloader, desc="-Validate", leave=False)
        for index, input, target in dataloader:
            self.val_step(index, input, target, run)
        run.on_val_end()

    def loop(self, run):
        it = range(self.epoch + 1, self.epoch + self.max_epochs + 1)
        if self.verbose == 1:
            it = tqdm(it)
        for self.epoch in it:
            run.on_epoch_start()
            self.train_loop(run)
            if run.dataloader.val is not None:
                self.val_loop(run)
            try:
                run.on_epoch_end()
            except StopIteration:
                break
            finally:
                if self.verbose:
                    tqdm.write(f"[{run.name}] epoch={self.epoch:03d} {run.metrics}")

    def fit(self, run):
        run.on_fit_start()
        try:
            self.loop(run)
        finally:
            run.on_fit_end()
