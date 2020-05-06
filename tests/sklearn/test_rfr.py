def test_rfr(client):
    experiment = client.create_experiment("rfr")
    run = experiment.create_run()
    run.start()

    assert repr(run.estimator).startswith('RandomForestRegressor(')
