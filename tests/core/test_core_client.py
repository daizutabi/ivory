def test_client_repr(client):
    assert repr(client) == "Client(num_objects=3)"


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
