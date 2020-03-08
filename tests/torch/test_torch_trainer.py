def test_trainer(run):
    trainer = run.trainer
    train_loader, val_loader = run.dataloaders[0]
    trainer.fit(train_loader, val_loader, run)
    assert trainer.epoch == 4
    assert run.metrics.score.shape == (5, 2)
    assert run.metrics.best_result.shape == (200, 2)
    assert run.metrics.best_epoch > -1
