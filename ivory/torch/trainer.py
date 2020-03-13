from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ivory.core.state import State
from ivory.torch import utils

try:
    from apex import amp
except ImportError:
    pass


@dataclass
class Trainer(State):
    fold: int = 0
    max_epochs: int = 1000
    gpu: bool = False
    precision: int = 32  # Full precision (32), half precision (16).
    amp_level: str = "O1"
    verbose: int = 1

    def __post_init__(self):
        self.epoch = -1
        self.global_step = -1

    def train_step(self, model, input):
        return model(input)

    def val_step(self, model, input):
        return model(input)

    def train(self, dataloader, metrics, model, optimizer):
        model.train()
        if self.verbose == 1:
            lr = optimizer.param_groups[0]["lr"]
            dataloader = tqdm(dataloader, desc=f"LR{lr:.1e}", leave=False)
        for index, input, target in dataloader:
            self.global_step += 1
            if self.gpu:
                input = utils.cuda(input)
                target = utils.cuda(target)
            output = self.train_step(model, input)
            loss = metrics.train_step(index, output, target)
            optimizer.zero_grad()
            if self.gpu and self.precision == 16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

    def val(self, dataloader, metrics, model):
        model.eval()
        if self.verbose == 1:
            dataloader = tqdm(dataloader, desc="-Validate", leave=False)
        with torch.no_grad():
            for index, input, target in dataloader:
                if self.gpu:
                    input = utils.cuda(input)
                    target = utils.cuda(target)
                output = self.val_step(model, input)
                metrics.val_step(index, output, target)

    def fit(self, run):
        train_loader, val_loader = run.dataloaders[self.fold]
        if self.gpu:
            run.model.cuda()
            if self.precision == 16:
                run.model, run.optimizer = amp.initialize(
                    run.model, run.optimizer, opt_level=self.amp_level
                )
        it = range(self.epoch + 1, self.epoch + self.max_epochs + 1)
        for self.epoch in tqdm(it) if self.verbose == 1 else it:
            run.on_epoch_start()
            self.train(train_loader, run.metrics, run.model, run.optimizer)
            self.val(val_loader, run.metrics, run.model)
            try:
                run.on_epoch_end()
            except StopIteration:
                break
            finally:
                if self.verbose:
                    tqdm.write(f"[{run.name}] epoch={self.epoch:03d} {run.metrics}")
            if 'scheduler' in run:
                if isinstance(run.scheduler, ReduceLROnPlateau):
                    run.scheduler.step(run.monitor.score)
                else:
                    run.scheduler.step()
