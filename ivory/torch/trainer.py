from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau

import ivory.core.trainer
from ivory.core import instance
from ivory.core.run import Run
from ivory.torch import utils

try:
    from apex import amp
except ImportError:
    pass


@dataclass
class Trainer(ivory.core.trainer.Trainer):
    loss: Optional[Callable] = None
    batch_size: int = 32
    shuffle: bool = True
    gpu: bool = False
    precision: int = 32  # Full precision (32), half precision (16).
    amp_level: str = "O1"
    scheduler_step_mode: str = "epoch"

    def __post_init__(self):
        self.loss = instance.get_attr(self.loss)

    def get_dataloader(self, run: Run, mode: str):
        shuffle = self.shuffle if mode == "train" else False
        return torch.utils.data.DataLoader(
            run.datasets[mode], batch_size=self.batch_size, shuffle=shuffle
        )

    def on_fit_begin(self, run):
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
        output = self.forward(run.model, input)
        if run.results:
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
        output = self.forward(run.model, input)
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
        self.on_fit_begin(run)
        run.model.eval()

    @torch.no_grad()
    def test_step(self, run, index, input, *target):
        if self.gpu:
            input = utils.cuda(input)
        output = self.forward(run.model, input)
        if run.results:
            run.results.step(index, output, *target)

    def forward(self, model, input):
        return model(input)
