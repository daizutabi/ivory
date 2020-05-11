from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import ivory.core.trainer
from ivory.torch import utils

try:
    from apex import amp
except ImportError:
    pass


@dataclass
class Trainer(ivory.core.trainer.Trainer):
    gpu: bool = False
    precision: int = 32  # Full precision (32), half precision (16).
    amp_level: str = "O1"
    scheduler_step_mode: str = "epoch"

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
        loss = run.metrics.step(input, output, target)
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
        run.metrics.step(input, output, target)

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
