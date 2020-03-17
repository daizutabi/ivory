def test_state(run):
    metrics = run.metrics
    state_dict = metrics.state_dict()
    assert "epoch" in state_dict
    assert metrics.epoch == metrics.epoch
    state_dict["epoch"] = 100
    metrics.load_state_dict(state_dict)
    assert metrics.epoch == 100
