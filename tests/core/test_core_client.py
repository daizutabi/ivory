import torch.nn

from ivory.core.client import create_client


def test_client_repr(client):
    assert repr(client) == "Client(num_objects=2)"


def test_search_run_ids(client):
    run = client.create_run("rfr")
    run.start("train")
    run.start("test")
    run_ids = list(client.search_run_ids("rfr"))
    x = len(run_ids)
    run = client.create_run("rfr")
    run.start("train")
    run_ids = list(client.search_run_ids("rfr"))
    assert len(run_ids) == x + 1

    task = client.create_task("rfr")
    a = task.create_run({"fold": 1})
    a.start("train")
    b = task.create_run({"fold": 2})
    b.start("train")

    run_ids = list(client.search_run_ids("rfr"))
    assert len(run_ids) == x + 4
    assert next(client.search_parent_run_ids("rfr")) == task.id
    assert len(list(client.search_nested_run_ids("rfr"))) == 2

    assert client.get_run_id("rfr", task=0) == task.id
    assert client.get_parent_run_id(a.id) == task.id
    assert next(client.search_run_ids("rfr", parent_run_id=task.id)) == b.id

    run = client.create_run("rfr")
    run.start("train")
    client.set_parent_run_id(run.id, task.id)
    assert len(list(client.search_run_ids("rfr", parent_run_id=task.id))) == 3


def test_create_task(client):
    task = client.create_task("rfr")
    name = task.name
    number = int(task.name.split("#")[1])
    task = client.create_task("rfr", number)
    assert task.name == name


def test_create_study(client):
    study = client.create_study("rfr")
    study.create_run({"fold": 1})
    assert len(list(client.search_run_ids("rfr", parent_only=True))) == 2
    assert client.create_study("rfr", -1).name == study.name


def test_create_evaluator(client):
    evaluator = client.create_evaluator()
    assert evaluator.client is client


def test_set_terminated(client):
    assert client.set_terminated("rfr") is None


def test_load(client):
    run = client.create_run("example")
    run.start()
    run_id = next(client.search_run_ids("example"))
    params = client.load_params(run_id)
    assert params["run"]["trainer"]["epochs"] == 10

    run = client.load_run(run_id)
    assert run.id == run_id

    trainer = client.load_instance(run_id, "trainer")
    assert trainer.epoch > 0


def test_load_instance(client, run):
    run = client.create_run("example")
    run.start("train")
    results = client.load_instance(run.id, "results", "test")
    assert "train" in results
    assert "val" in results
    assert "test" not in results
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


def test_remove_delted_runs(client):
    run = client.create_run("rfr")
    run.start()
    client.tracker.client.delete_run(run.id)
    assert client.remove_deleted_runs() == 1
