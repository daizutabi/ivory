from dataclasses import dataclass

from tqdm import tqdm

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

    def loop(self, run):
        max_epoch = self.epoch + self.max_epochs
        width = len(str(max_epoch))
        it = range(self.epoch + 1, max_epoch + 1)
        if self.verbose == 1:
            it = tqdm(it, desc="Epoch ")
        for self.epoch in it:
            run.on_epoch_start()
            self.train_loop(run)
            if run.dataloaders.val:
                self.val_loop(run)
            try:
                run.on_epoch_end()
            except StopIteration:
                break
            finally:
                if self.verbose:
                    epoch = str(self.epoch).zfill(width)
                    tqdm.write(f"[{run.name}] epoch={epoch} {run.metrics}")

    def fit(self, run):
        run.on_fit_start()
        try:
            self.loop(run)
        finally:
            run.on_fit_end()

    def test(self, run):
        run.on_test_start()
        dataloader = run.dataloaders.test
        if self.verbose == 1:
            dataloader = tqdm(dataloader, desc="Test ", leave=False)
        for index, input in dataloader:
            self.test_step(index, input, run)
        run.on_test_end()

    def test_step(self, index, input, run):
        pass
