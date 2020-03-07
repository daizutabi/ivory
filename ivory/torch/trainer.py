from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm

from ivory.callback import CallbackCaller
from ivory.torch.utils import cuda

try:
    from apex import amp
except ImportError:
    pass


@dataclass
class Trainer(CallbackCaller):
    epoch: int = -1
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

    def fit(self, train_loader, val_loader, cfg):
        if self.gpu:
            cfg.model.cuda()
            if self.amp_level:
                cfg.model, cfg.optimizer = amp.initialize(
                    cfg.model, cfg.optimizer, opt_level=self.amp_level
                )

        self.on_fit_start(cfg)
        it = range(self.epoch + 1, self.epoch + self.max_epochs + 1)
        with tqdm(it) as t:
            for self.epoch in t:
                t.set_description(f"epoch={self.epoch:03d}")
                self.on_epoch_start(cfg)
                self.on_train_start(cfg)
                self.train(train_loader, cfg.metrics, cfg.model, cfg.optimizer)
                self.on_train_end(cfg)
                self.on_val_start(cfg)
                self.val(val_loader, cfg.metrics, cfg.model)
                self.on_val_end(cfg)
                if cfg.scheduler:
                    cfg.scheduler.step()
                try:
                    self.on_epoch_end(cfg)
                except StopIteration:
                    t.set_description("Stopped")
                    break
                finally:
                    tqdm.write(f"epoch={self.epoch:03d} {cfg.metrics}")
                if self.epoch == self.max_epochs - 1:
                    t.set_description("Finished")
            self.on_fit_end(cfg)
