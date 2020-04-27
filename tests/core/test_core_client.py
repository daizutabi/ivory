import pytest

from ivory.core.client import create_client


def test_client_repr(client):
    assert repr(client) == "Client(num_objects=1)"


def test_create_params(client):
    params = client.create_params("example")
    assert params["experiment"]["name"] == "example"


def test_get_experiment(client):
    experiment = next(client.get_experiments())
    assert experiment.name == "example"
    for experiment in client.get_experiments():
        pass
    assert experiment.name == "example"


def test_search_runs(client, run):
    run.start("train")
    run.start("test")
    run_ids = list(client.search_runs())
    assert len(run_ids)


def test_get_experiment_from_run_id(client):
    run_ids = list(client.search_runs())
    experiment = client.get_experiment_from_run_id(run_ids[0])
    assert experiment.name == "example"

    with pytest.raises(ValueError):
        client.get_experiment_from_run_id("ABC")


def test_load(client):
    run_id = next(client.search_runs())
    params = client.load_params(run_id)
    assert params["run"]["trainer"]["max_epochs"] == 10

    run = client.load_run(run_id)
    assert run.id == run_id

    trainer = client.load_instance(run_id, "trainer")
    assert trainer.epoch > 0

    output, target = client.load_results([run_id, run_id])
    assert len(output) == 200 * 2 * 2


def test_without_tracker():
    client = create_client(directory="tests", tracker=False)
    assert "tracker" not in client
