import torch.nn


def test_experiment_create_run(experiment, params):
    run = experiment.create_run()
    assert run.id
    assert run.dataloaders.fold == 0
    params["run"]["dataloaders"]["fold"] = 4
    run = experiment.create_run(params)
    assert run.dataloaders.fold == 4


def test_load_run(experiment, run):
    run.start()
    run = experiment.load_run(run.id, "best")
    assert run.trainer.epoch != -1


def test_load_instance(experiment, run):
    results = experiment.load_instance(run.id, "results", "test")
    assert "train" in results
    assert "val" in results
    assert "test" in results
    model = experiment.load_instance(run.id, "model", "test")
    assert isinstance(model, torch.nn.Module)
