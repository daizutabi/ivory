def test_trainer(run):
    trainer = run.trainer
    trainer.fit(run)
    assert trainer.epoch == 4
    assert run.metrics.history.shape == (5, 2)
    assert run.metrics.best_output.shape == (200, 1)
    assert run.metrics.best_epoch > -1
