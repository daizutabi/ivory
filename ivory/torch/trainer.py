from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ivory.torch.utils import cuda

try:
    from apex import amp
except ImportError:
    pass


@dataclass
class Trainer:
    fold: int = 0
    epoch: int = -1
    global_step: int = -1
    max_epochs: int = 1000
    gpu: bool = False
    amp_level: Optional[str] = None
    verbose: int = 1

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
                input = cuda(input)
                target = cuda(target)
            output = self.train_step(model, input)
            loss = metrics.train_step(index, output, target)
            optimizer.zero_grad()
            if self.gpu and self.amp_level:
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
                    input = cuda(input)
                    target = cuda(target)
                output = self.val_step(model, input)
                metrics.val_step(index, output, target)

    def fit(self, run):
        train_loader, val_loader = run.dataloaders[self.fold]
        if self.gpu:
            run.model.cuda()
            if self.amp_level:
                run.model, run.optimizer = amp.initialize(
                    run.model, run.optimizer, opt_level=self.amp_level
                )
        it = range(self.epoch + 1, self.epoch + self.max_epochs + 1)
        for self.epoch in tqdm(it) if self.verbose == 1 else it:
            run.on_epoch_start()
            run.on_train_start()
            self.train(train_loader, run.metrics, run.model, run.optimizer)
            run.on_train_end()
            run.on_val_start()
            self.val(val_loader, run.metrics, run.model)
            run.on_val_end()
            try:
                run.on_epoch_end()
            except StopIteration:
                break
            finally:
                if self.verbose:
                    latest = run.metrics.latest
                    tqdm.write(f"[{run.name}] epoch={self.epoch:03d} {latest}")
            if run.scheduler:
                if isinstance(run.scheduler, ReduceLROnPlateau):
                    run.scheduler.step(run.metrics.current_score)
                else:
                    run.scheduler.step()

    def state_dict(self):
        return {
            "fold": self.fold,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.fold = state_dict["fold"]
        self.epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
