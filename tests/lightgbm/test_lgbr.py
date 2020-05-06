def test_lgbr(client):
    experiment = client.create_experiment("lgbr")
    run = experiment.create_run()
    run.start()

    assert repr(run.estimator).startswith('Regressor(params=')
