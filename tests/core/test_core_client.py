import torch


def test_client_repr(client):
    assert repr(client) == "Client(name='params', num_objects=3)"


def test_client_create_run(client, params):
    run = client.create_run()
    assert run.id
    assert run.dataloaders.fold == 0
    params["run"]["dataloaders"]["fold"] = 4
    run = client.create_run(params)
    assert run.dataloaders.fold == 4


def test_client_create_instance(client):
    data = client.create_instance("experiment.data")
    assert data.num_samples == 1000


def test_client_run_str(client):
    for run in client.run(["fold=1"], message="test"):
        assert run.dataloaders.fold == 1
        assert run.name == "single"
    for run in client.run(["fold=1", "lr=1e-3"], message="test"):
        assert run.optimizer.param_groups[0]["lr"] == 1e-3
        assert run.name == "single"
    for k, run in enumerate(client.run(["fold=1,2"], message="test")):
        if k == 0:
            assert run.dataloaders.fold == 1
            assert run.name == "scan#1"
        if k == 1:
            assert run.dataloaders.fold == 2
            assert run.name == "scan#2"
    for k, run in enumerate(client.run(["fold=1,2", "max_epochs=3,4"])):
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
    for k, run in enumerate(client.run(["fold=1"], repeat=3)):
        assert run.name == f"repeat#{k + 1}"


def test_client_run_kwargs(client):
    for k, run in enumerate(client.run(fold="1-2", max_epochs="3,4")):
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


def test_optimize_lr(client):
    study = next(client.optimize("lr", {"fold": 2, "batch_size": 10}, n_trials=3))
    assert study.user_attrs == {"experiment_id": client.experiment.id}
    trials = study.trials
    assert len(trials) == 3
    assert "run_id" in trials[0].user_attrs


def test_optimize_hidden_sizes(client):
    study = next(
        client.optimize("hidden_sizes", {"fold": 1, "batch_size": 20}, n_trials=4)
    )
    assert study.user_attrs == {"experiment_id": client.experiment.id}


def test_search_runs(client):
    runs = list(client.search_runs())
    assert len(runs) >= 0
    runs = list(client.search_runs(tags={"batch_size": "10"}))
    assert len(runs) == 3
    runs = list(client.search_runs(tags={"batch_size": "20"}))
    assert len(runs) == 4


def test_load_run(client, run):
    run.start()
    run = client.load_run(run.id, "best")
    assert run.trainer.epoch != -1

    runs = list(client.load_runs([run.id], "test"))
    assert len(runs) == 1


def test_load_instance(client, run):
    results = client.load_instance(run.id, "results", "best")
    assert "train" in results
    assert "test" not in results
    model = client.load_instance(run.id, "model", "test")
    assert isinstance(model, torch.nn.Module)

    results = list(client.load_instances([run.id], "results", "best"))
    assert len(results) == 1


# def suggest_lr(trial):
#     trial.suggest_loguniform("lr", 1e-4, 1e-1)
#
#
# def suggest_hidden_sizes(trial):
