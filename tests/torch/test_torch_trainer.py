def test_trainer(cfg):
    trainer = cfg.trainer
    train_loader, val_loader = cfg.dataloaders[0]
    trainer.fit(train_loader, val_loader, cfg)
    assert trainer.epoch == 4
    assert cfg.metrics.score.shape == (5, 2)
    assert cfg.metrics.best_result.shape == (200, 2)
    assert cfg.metrics.best_epoch > -1
