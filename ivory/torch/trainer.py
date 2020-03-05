from dataclasses import dataclass
from typing import Optional

import torch
from apex import amp
from tqdm import tqdm


@dataclass
class Trainer:
    epoch: int = 0
    max_epochs: int = 1000
    train_percent_check: float = 1.0
    val_percent_check: float = 1.0
    amp_level: Optional[str] = None

    def train(self, dataloader, metrics, model, optimizer):
        model.train()
        metrics.reset()
        total = int(self.train_percent_check * len(dataloader))
        for _, batch in tqdm(zip(range(total), dataloader), total=total):
            index, input, target = batch
            output = self.train_step(model, input)
            loss = metrics.step(index, output, target)
            optimizer.zero_grad()
            if self.amp_level:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

    def validate(self, dataloader, metrics, model):
        model.eval()
        metrics.reset()
        total = int(self.val_percent_check * len(dataloader))
        with torch.no_grad():
            for _, batch in tqdm(zip(range(total), dataloader), total=total):
                index, input, target = batch
                output = self.validate_step(model, input)
                metrics.step(index, output, target)

    def train_end(self, metrics):
        pass

    def validate_end(self, metrics):
        pass

    def fit(self, train_loader, val_loader, cfg):
        if self.amp_level:
            cfg.model, cfg.optimizer = amp.initialize(
                cfg.model, cfg.optimizer, opt_level=self.amp_level
            )

        for epoch in range(0, self.max_epochs):
            # lr = self.optimizer.param_groups[0]["lr"]
            self.train(train_loader, cfg.metrics, cfg.model, cfg.optimizer)
            self.train_end(cfg.metrics)
            self.validate(val_loader, cfg.metrics, cfg.model)
            self.validate_end(cfg.metrics)
            if cfg.scheduler:
                cfg.scheduler.step()
            self.epoch += 1
