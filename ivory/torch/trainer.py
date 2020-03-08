from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange

from ivory.core.callback import CallbackCaller
from ivory.torch.utils import cuda

try:
    from apex import amp
except ImportError:
    pass


@dataclass
class Trainer(CallbackCaller):
    epoch: int = -1
    global_step: int = -1
    max_epochs: int = 1000
    gpu: bool = False
    amp_level: Optional[str] = None

    def train_step(self, model, input):
        return model(input)

    def val_step(self, model, input):
        return model(input)

    def train(self, dataloader, metrics, model, optimizer):
        model.train()
        lr = optimizer.param_groups[0]["lr"]
        it = tqdm(dataloader, desc=f"LR{lr:.1e}", leave=False)
        for index, input, target in it:
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
        with torch.no_grad():
            it = tqdm(dataloader, desc="-Validate", leave=False)
            for index, input, target in it:
                if self.gpu:
                    input = cuda(input)
                    target = cuda(target)
                output = self.val_step(model, input)
                metrics.val_step(index, output, target)

    def fit(self, train_loader, val_loader, run):
        if self.gpu:
            run.model.cuda()
            if self.amp_level:
                run.model, run.optimizer = amp.initialize(
                    run.model, run.optimizer, opt_level=self.amp_level
                )
        with trange(self.epoch + 1, self.epoch + self.max_epochs + 1) as t:
            self.on_fit_start(run)
            for self.epoch in t:
                t.set_description(f"epoch={self.epoch:03d}")
                self.on_epoch_start(run)
                self.on_train_start(run)
                self.train(train_loader, run.metrics, run.model, run.optimizer)
                self.on_train_end(run)
                self.on_val_start(run)
                self.val(val_loader, run.metrics, run.model)
                self.on_val_end(run)
                try:
                    self.on_epoch_end(run)
                except StopIteration:
                    t.set_description("Stopped")
                    break
                finally:
                    lr = run.optimizer.param_groups[0]["lr"]
                    latest = run.metrics.latest
                    tqdm.write(f"epoch={self.epoch:03d} lr={lr:.1e} {latest}")
                if run.scheduler:
                    if isinstance(run.scheduler, ReduceLROnPlateau):
                        run.scheduler.step(run.metrics.current_score)
                    else:
                        run.scheduler.step()
                if self.epoch == self.max_epochs - 1:
                    t.set_description("Finished")
            self.on_fit_end(run)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
