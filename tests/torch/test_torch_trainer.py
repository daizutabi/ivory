def test_trainer(obj):
    trainer = obj.trainer
    train_loader, val_loader = obj.dataloaders[0]
    trainer.fit(train_loader, val_loader, obj)
    assert trainer.epoch == 4
    assert obj.metrics.score.shape == (5, 2)
    assert obj.metrics.best_result.shape == (200, 2)
    assert obj.metrics.best_epoch > -1
