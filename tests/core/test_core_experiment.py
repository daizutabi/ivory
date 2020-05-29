def test_experiment_create_run(experiment, params):
    run = experiment.create_run()
    assert run.id
    assert run.datasets.fold == 0
