# Trainer

Ivory's `ivory.torch.Trainer` instance manages train and validation loop. At this stage, just take a look at a simple version of the code.

```python
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class Trainer:
    fold: int = 0
    epoch: int = -1
    global_step: int = -1
    max_epochs: int = 1000

    def train_step(self, model, input):
        return model(input)

    def val_step(self, model, input):
        return model(input)

    def train(self, dataloader, metrics, model, optimizer):
        model.train()
        for index, input, target in dataloader:
            self.global_step += 1
            output = self.train_step(model, input)
            loss = metrics.train_step(index, output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def val(self, dataloader, metrics, model):
        model.eval()
        with torch.no_grad():
            for index, input, target in dataloader:
                output = self.val_step(model, input)
                metrics.val_step(index, output, target)

    def fit(self, run):
        train_loader, val_loader = run.dataloaders[self.fold]
        it = range(self.epoch + 1, self.epoch + self.max_epochs + 1)
        for self.epoch in it:
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
            if run.scheduler:
                if isinstance(run.scheduler, ReduceLROnPlateau):
                    run.scheduler.step(run.metrics.current_score)
                else:
                    run.scheduler.step()
```
