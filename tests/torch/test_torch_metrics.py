import numpy as np
import torch


def step(func, index, target, x):
    for i in range(3):
        func(index, x * target, target)


def test_metrics(run):
    metrics = run.metrics
    metrics.on_epoch_start(run)
    train_loader, val_loader = run.dataloaders[0]
    it = iter(train_loader)
    index, input, target = next(it)
    loss = metrics.train_step(index, 1.02 * target, target)
    assert np.allclose(loss.item(), torch.mean((0.02 * target) ** 2).item())
    assert metrics.train_batch_loss[0] == loss.item()
    step(metrics.train_step, index, target, 1.02)
    step(metrics.val_step, index, target, 1.04)
    assert len(metrics.train_batch_loss) == 4
    assert len(metrics.val_batch_loss) == 3

    metrics.on_epoch_end(run)
    history = metrics.history
    assert len(history) == 1
    assert list(history[metrics.epoch].keys()) == ["loss", "val_loss", "lr"]
    assert 'num_records=1' in repr(metrics)
