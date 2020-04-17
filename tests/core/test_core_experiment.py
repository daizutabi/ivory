import torch.nn


def test_experiment_create_run(experiment, params):
    run = experiment.create_run()
    assert run.id
    assert run.dataloaders.fold == 0
    params["run"]["dataloaders"]["fold"] = 4
    run = experiment.create_run(params)
    assert run.dataloaders.fold == 4


# def test_experiment_create_instance(experiment):
#     data = experiment.create_instance("experiment.data")
#     assert data.num_samples == 1000


def test_experiment_run_str(experiment):
    for run in experiment.start(["fold=1"], message="test"):
        assert run.dataloaders.fold == 1
        assert run.name == "single"
    for run in experiment.start(["fold=1", "lr=1e-3"], message="test"):
        assert run.optimizer.param_groups[0]["lr"] == 1e-3
        assert run.name == "single"
    for k, run in enumerate(experiment.start(["fold=1,2"], message="test")):
        if k == 0:
            assert run.dataloaders.fold == 1
            assert run.name == "scan#1"
        if k == 1:
            assert run.dataloaders.fold == 2
            assert run.name == "scan#2"
    for k, run in enumerate(experiment.start(["fold=1,2", "max_epochs=3,4"])):
        if k == 0:
            assert run.dataloaders.fold == 1
            assert run.trainer.max_epochs == 3
            assert run.name == "prod#1"
        if k == 1:
            assert run.dataloaders.fold == 1
            assert run.trainer.max_epochs == 4
            assert run.name == "prod#2"
        if k == 2:
            assert run.dataloaders.fold == 2
            assert run.trainer.max_epochs == 3
            assert run.name == "prod#3"
        if k == 3:
            assert run.dataloaders.fold == 2
            assert run.trainer.max_epochs == 4
            assert run.name == "prod#4"
    for k, run in enumerate(experiment.start(["fold=1"], repeat=3)):
        assert run.name == f"repeat#{k + 1}"


def test_experiment_run_kwargs(experiment):
    for k, run in enumerate(experiment.start(fold="1-2", max_epochs="3,4")):
        if k == 0:
            assert run.dataloaders.fold == 1
            assert run.trainer.max_epochs == 3
            assert run.name == "prod#1"
        if k == 1:
            assert run.dataloaders.fold == 1
            assert run.trainer.max_epochs == 4
            assert run.name == "prod#2"
        if k == 2:
            assert run.dataloaders.fold == 2
            assert run.trainer.max_epochs == 3
            assert run.name == "prod#3"
        if k == 3:
            assert run.dataloaders.fold == 2
            assert run.trainer.max_epochs == 4
            assert run.name == "prod#4"


def test_optimize_lr(experiment):
    study = next(
        experiment.optimize("suggest_lr", {"fold": 2, "batch_size": 10}, n_trials=3)
    )
    assert study.user_attrs == {"experiment_id": experiment.id}
    trials = study.trials
    assert len(trials) == 3
    assert "run_id" in trials[0].user_attrs


def test_optimize_hidden_sizes(experiment):
    study = next(
        experiment.optimize(
            "suggest_hidden_sizes", {"fold": 1, "batch_size": 20}, n_trials=4
        )
    )
    assert study.user_attrs == {"experiment_id": experiment.id}


def test_load_run(experiment, run):
    run.start()
    run = experiment.load_run(run.id, "best")
    assert run.trainer.epoch != -1


def test_load_instance(experiment, run):
    results = experiment.load_instance(run.id, "results", "best")
    assert "train" in results
    assert "test" not in results
    model = experiment.load_instance(run.id, "model", "test")
    assert isinstance(model, torch.nn.Module)
