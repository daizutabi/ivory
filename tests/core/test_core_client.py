def test_client_repr(client):
    assert repr(client) == "Client(num_objects=3)"


def test_client_create_run(client, params):
    run = client.create_run()
    assert run.id
    assert run.dataloader.fold == 0
    params["run"]["dataloader"]["fold"] = 4
    run = client.create_run(params)
    assert run.dataloader.fold == 4


def test_client_create_instance(client):
    data = client.create_instance("experiment.data")
    assert data.num_samples == 1000


def test_client_product(client):
    for run in client.product(["fold=1"], message="test"):
        assert run.dataloader.fold == 1
        assert run.name == "single"
    for run in client.product(["fold=1", "lr=1e-3"], message="test"):
        assert run.optimizer.param_groups[0]["lr"] == 1e-3
        assert run.name == "single"
    for k, run in enumerate(client.product(["fold=1,2"], message="test")):
        if k == 0:
            assert run.dataloader.fold == 1
            assert run.name == "scan#1"
        if k == 1:
            assert run.dataloader.fold == 2
            assert run.name == "scan#2"
    for k, run in enumerate(client.product(["fold=1,2", "max_epochs=3,4"])):
        if k == 0:
            assert run.dataloader.fold == 1
            assert run.trainer.max_epochs == 3
            assert run.name == "product#1"
        if k == 1:
            assert run.dataloader.fold == 1
            assert run.trainer.max_epochs == 4
            assert run.name == "product#2"
        if k == 2:
            assert run.dataloader.fold == 2
            assert run.trainer.max_epochs == 3
            assert run.name == "product#3"
        if k == 3:
            assert run.dataloader.fold == 2
            assert run.trainer.max_epochs == 4
            assert run.name == "product#4"
    for k, run in enumerate(client.product(["fold=1"], repeat=3)):
        assert run.name == f"repeat#{k + 1}"


def test_client_chain(client):
    for k, run in enumerate(client.chain(["fold=1-2", "max_epochs=3,4"])):
        if k == 0:
            assert run.dataloader.fold == 1
            assert run.trainer.max_epochs == 10
            assert run.name == "chain#1"
        if k == 1:
            assert run.dataloader.fold == 2
            assert run.trainer.max_epochs == 10
            assert run.name == "chain#2"
        if k == 2:
            assert run.dataloader.fold == 0
            assert run.trainer.max_epochs == 3
            assert run.name == "chain#3"
        if k == 3:
            assert run.dataloader.fold == 0
            assert run.trainer.max_epochs == 4
            assert run.name == "chain#4"