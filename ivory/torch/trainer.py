from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau

import ivory.core.trainer
import ivory.torch.data
import ivory.torch.functions
from ivory.core import instance
from ivory.torch import utils

try:
    from apex import amp
except ImportError:
    pass


@dataclass
class Trainer(ivory.core.trainer.Trainer):
    loss: Optional[Callable] = None
    dataloaders: str = "ivory.torch.data.DataLoaders"
    gpu: bool = False
    precision: int = 32  # Full precision (32), half precision (16).
    amp_level: str = "O1"
    scheduler_step_mode: str = "epoch"

    def __post_init__(self):
        if isinstance(self.loss, str) and "." not in self.loss:
            self.loss = getattr(ivory.torch.functions, self.loss)
        else:
            self.loss = instance.get_attr(self.loss)

    def on_init_begin(self, run):
        super().on_init_begin(run)
        if self.gpu:
            run.model.cuda()
            if self.precision == 16:
                run.model, run.optimizer = amp.initialize(
                    run.model, run.optimizer, opt_level=self.amp_level
                )

    def on_train_begin(self, run):
        run.model.train()

    def train_step(self, run, index, input, target):
        if self.gpu:
            input = utils.cuda(input)
            target = utils.cuda(target)
        output = run.model(input)
        run.results.step(index, output, target)
        loss = self.loss(output, target)
        run.metrics.step(loss.item())
        optimizer = run.optimizer
        optimizer.zero_grad()
        if self.gpu and self.precision == 16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if run.sheduler and self.scheduler_step_mode == "batch":
            run.scheduler.step()

    def on_val_begin(self, run):
        run.model.eval()

    @torch.no_grad()
    def val_step(self, run, index, input, target):
        if self.gpu:
            input = utils.cuda(input)
            target = utils.cuda(target)
        output = run.model(input)
        if run.results:
            run.results.step(index, output, target)
        loss = self.loss(output, target)
        run.metrics.step(loss.item())

    def on_epoch_end(self, run):
        if run.scheduler and self.scheduler_step_mode == "epoch":
            if isinstance(run.scheduler, ReduceLROnPlateau):
                run.scheduler.step(run.monitor.score)
            else:
                run.scheduler.step()

    def on_test_begin(self, run):
        run.model.eval()

    @torch.no_grad()
    def test_step(self, run, index, input, *target):
        if self.gpu:
            input = utils.cuda(input)
        output = run.model(input)
        if run.results:
            run.results.step(index, output, *target)
