def test_optimize_lr(study, client):
    s = study.optimize("lr", n_trials=3, fold=2)
    assert s.user_attrs == {"run_id": study.id}
    trials = s.trials
    assert len(trials) == 3
    assert "run_id" in trials[0].user_attrs
    run_id = trials[0].user_attrs["run_id"]
    params = client.load_params(run_id)
    assert params["run"]["datasets"]["fold"] == 2


def test_optimize_hidden_sizes(study, client):
    s = study.optimize("hidden_sizes", n_trials=4)
    assert s.user_attrs == {"run_id": study.id}
