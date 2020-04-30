import torch.nn

from ivory.core.client import create_client


def test_client_repr(client):
    assert repr(client) == "Client(num_objects=1)"


def test_search_run_ids(client, run):
    run.start("train")
    run.start("test")
    run_ids = list(client.search_run_ids())
    assert len(run_ids)


def test_load(client):
    run_id = next(client.search_run_ids())
    params = client.load_params(run_id)
    assert params["run"]["trainer"]["max_epochs"] == 10

    run = client.load_run(run_id)
    assert run.id == run_id

    trainer = client.load_instance(run_id, "trainer")
    assert trainer.epoch > 0

    output, target = client.load_results([run_id, run_id])
    assert len(output) == 200 * 2 * 2


def test_load_instance(client, run):
    results = client.load_instance(run.id, "results", "test")
    assert "train" in results
    assert "val" in results
    assert "test" in results
    model = client.load_instance(run.id, "model", "test")
    assert isinstance(model, torch.nn.Module)


def test_without_tracker():
    client = create_client(directory="tests", tracker=False)
    assert "tracker" not in client


def test_update_params(client, experiment):
    run = experiment.create_run(fold=0)
    run = experiment.create_run(args={"hidden_sizes.0": 10})
    assert "fold" not in run.tracking.client.get_run(run.id).data.params
    client.update_params(experiment.name)
    assert "fold" in run.tracking.client.get_run(run.id).data.params
