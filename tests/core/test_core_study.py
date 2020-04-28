def test_optimize_lr(study):
    s = study.optimize("lr", n_trials=3)
    assert s.user_attrs == {"run_id": study.id}
    trials = s.trials
    assert len(trials) == 3
    assert "run_id" in trials[0].user_attrs


def test_optimize_hidden_sizes(study):
    s = study.optimize("hidden_sizes", n_trials=4)
    assert s.user_attrs == {"run_id": study.id}
