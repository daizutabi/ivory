"""The `ivory.nnabla.trainer` module provides the `Trainer` class for nnable."""
from dataclasses import dataclass
from typing import Callable, Optional

import nnabla as nn
from nnabla.ext_utils import get_extension_context

import ivory.core.trainer
import ivory.nnabla.data
import ivory.nnabla.functions
from ivory.core import instance


@dataclass
class Trainer(ivory.core.trainer.Trainer):
    loss: Optional[Callable] = None
    dataloaders: str = "ivory.nnabla.data.DataLoaders"
    gpu: bool = False
    precision: int = 32  # Full precision (32), half precision (16).
    amp_level: str = "O1"

    def __post_init__(self):
        if isinstance(self.loss, str) and "." not in self.loss:
            self.loss = getattr(ivory.nnabla.functions, self.loss)
        else:
            self.loss = instance.get_attr(self.loss)

    def on_init_begin(self, run):
        super().on_init_begin(run)
        if self.gpu:
            context = "cudnn"
        else:
            context = "cpu"
        if self.precision == 32:
            type_config = "float"
        elif self.precision == 16:
            type_config = "half"
        else:
            raise ValueError(f"Unknown precision: {self.precision}")

        context = get_extension_context(context, type_config=type_config)
        nn.set_default_context(context)

        if not run.model.parameters():
            run.model.build(self.loss, run.datasets.train, self.batch_size)
            run.optimizer.set_parameters(run.model.parameters())

    def on_train_begin(self, run):
        run.model.train()

    def train_step(self, run, index, input, target):
        optimizer = run.optimizer
        optimizer.zero_grad()
        output, loss = run.model(input, target)
        run.results.step(index, output, target)
        run.metrics.step(loss)
        run.model.backward()
        optimizer.update()

    def on_val_begin(self, run):
        run.model.eval()

    def val_step(self, run, index, input, target):
        output, loss = run.model(input, target)
        run.results.step(index, output, target)
        run.metrics.step(loss)

    def on_epoch_end(self, run):
        pass

    def on_test_begin(self, run):
        run.model.eval()

    def test_step(self, run, index, input, target):
        output = run.model(input)
        run.results.step(index, output, target)
