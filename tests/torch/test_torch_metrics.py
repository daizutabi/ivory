import numpy as np
import torch
import torch.nn.functional as F


def step(func, index, target, x):
    for i in range(3):
        func(index, x * target, target)


def test_metrics(metrics, run):
    assert metrics.criterion is F.mse_loss

    metrics.on_epoch_start(run)
    metrics.on_val_start(run)

    train_loader, val_loader = run.dataloaders[0]
    it = iter(train_loader)
    index, input, target = next(it)
    loss = metrics.train_step(index, 1.02 * target, target)
    assert np.allclose(loss.item(), torch.mean((0.02 * target) ** 2).item())
    assert metrics.train_batch_record[0] == {"loss": loss.item()}
    step(metrics.train_step, index, target, 1.02)
    step(metrics.val_step, index, target, 1.04)
    assert len(metrics.train_batch_record) == 4
    assert len(metrics.val_batch_record) == 3

    metrics.on_epoch_end(run)
    history = metrics.history
    assert len(history) == 1
    assert history.columns.tolist() == ["loss", "val_loss", "lr"]
    assert np.allclose(history["loss"].iloc[0], torch.mean((0.02 * target) ** 2).item())
    assert np.allclose(
        history["val_loss"].iloc[0], torch.mean((0.04 * target) ** 2).item()
    )

    step(metrics.val_step, index, target, 1.05)
    metrics.on_epoch_end(run)
    step(metrics.val_step, index, target, 1.02)
    metrics.on_epoch_end(run)
    step(metrics.val_step, index, target, 1.03)
    metrics.on_epoch_end(run)
    assert len(metrics.history) == 4
    metrics.history.val_loss.argmin() == 2
