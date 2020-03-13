import numpy as np


def test_trainer(run):
    run.start()
    assert run.trainer.epoch == 4
    assert len(run.metrics.history) == 5
    assert run.metrics.data['index'].shape == (200,)
    assert run.metrics.data['output'].shape == (200, 1)
    assert run.monitor.best_epoch > -1

    state_dict = run.state_dict()
    assert state_dict["trainer"]["epoch"] == 4
    assert state_dict["trainer"]["global_step"] == 399
    assert state_dict["early_stopping"]["best_score"] > 0

    run.start()
    assert run.trainer.epoch == 9

    run.load_state_dict(state_dict)
    assert run.trainer.epoch == 4

    run.early_stopping.mode = "max"
    run.metrics.mode = "max"
    run.early_stopping.best_score = np.inf
    run.early_stopping.patience = -1
    run.start()
    assert run.early_stopping.wait == 1
    assert run.trainer.epoch == 5
