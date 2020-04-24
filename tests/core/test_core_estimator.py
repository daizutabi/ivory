def test_estimator(client):
    experiment = client.create_experiment("ridge")
    assert experiment.name == "ridge"

    run = experiment.create_run()
    run.start("train")
    assert run.results.train["output"].shape == (600, 1)
    assert run.results.val["output"].shape == (200, 1)
    run.start("test")
    assert run.results.test["output"].shape == (200, 1)
