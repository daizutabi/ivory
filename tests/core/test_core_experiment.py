def test_experiment_create_run(experiment, params):
    run = experiment.create_run()
    assert run.id
    assert run.dataloaders.fold == 0
    params["run"]["dataloaders"]["fold"] = 4
    run = experiment.create_run(params)
    assert run.dataloaders.fold == 4


def test_update_params(experiment):
    run = experiment.create_run(fold=0)
    run.start()
    run = experiment.create_run(args={"hidden_sizes.0": 10})
    run.start()
    assert "fold" not in run.tracking.client.get_run(run.id).data.params
    experiment.update_params()
    assert "fold" in run.tracking.client.get_run(run.id).data.params
