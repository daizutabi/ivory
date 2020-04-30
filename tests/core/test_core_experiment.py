def test_experiment_create_run(experiment, params):
    run = experiment.create_run()
    assert run.id
    assert run.dataloaders.fold == 0
