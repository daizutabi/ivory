import numpy as np
import torch


def step(func, target, x):
    for i in range(3):
        func(x * target, target)


def test_metrics(metrics, dataloaders, data, run):
    dataloaders.init(data)
    metrics.on_epoch_start(run)
    metrics.on_train_start(run)
    train_loader = dataloaders.train
    it = iter(train_loader)
    index, input, target = next(it)
    loss = metrics.step(1.02 * target, target)
    assert np.allclose(loss.item(), torch.mean((0.02 * target) ** 2).item())
    assert metrics.losses[0] == loss.item()
    step(metrics.step, target, 1.02)
    assert len(metrics.losses) == 4
    metrics.on_train_end(run)
    assert metrics.loss > 0
    metrics.on_val_start(run)
    train_loader = dataloaders.train
    it = iter(train_loader)
    index, input, target = next(it)
    loss = metrics.step(1.02 * target, target)
    metrics.on_val_end(run)
    metrics.on_epoch_end(run)
    history = metrics.history
    assert list(history.keys()) == ["loss", "val_loss", "lr"]
    assert "(loss=" in repr(metrics)
