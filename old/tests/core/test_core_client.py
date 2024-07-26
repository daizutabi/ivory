import numpy as np
import torch.nn

from ivory.core.client import create_client


def test_client_repr(client):
    assert repr(client) == "Client(num_instances=2)"


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
    a_number = int(a.name.split("#")[1])
    b_number = int(b.name.split("#")[1])
    assert list(client.get_run_ids("rfr", run=[a_number, b_number])) == [a.id, b.id]
    assert client.get_parent_run_id("rfr", run=a_number) == task.id
    assert list(client.get_nested_run_ids("rfr", task=0)) == [b.id, a.id]
    assert next(client.search_run_ids("rfr", parent_run_id=task.id)) == b.id

    run = client.create_run("rfr")
    run.start("train")
    run_number = int(run.name.split("#")[1])
    client.set_parent_run_id("rfr", run=run_number, task=0)
    assert len(list(client.search_run_ids("rfr", parent_run_id=task.id))) == 3
    e = run

    task = client.create_task("rfr")
    c = task.create_run({"fold": 1})
    c.start("train")
    d = task.create_run({"fold": 2})
    d.start("train")

    run_ids = client.get_nested_run_ids("rfr", task=[1, 0])
    assert list(run_ids) == [d.id, c.id, e.id, b.id, a.id]
    run_ids = client.get_nested_run_ids("rfr", task=[0, 1], fold=1)
    assert list(run_ids) == [a.id, c.id]


def test_search_run_ids_best_score(client):
    a = client.create_run("example")
    a.start()
    b = client.create_run("example")
    b.start()
    best_score = np.inf
    for run_id in client.search_run_ids("example", exclude_parent=True):
        monitor = client.load_instance(run_id, "monitor")
        best_score = min(best_score, monitor.best_score)

    run_ids = client.search_run_ids(
        "example", best_score_limit=best_score, exclude_parent=True
    )
    assert len(list(run_ids)) == 1


def test_create_task(client):
    task = client.create_task("rfr")
    name = task.name
    number = int(task.name.split("#")[1])
    task = client.create_task("rfr", number)
    assert task.name == name


def test_create_study(client):
    study = client.create_study("rfr")
    study.create_run({"fold": 1})
    assert len(list(client.search_run_ids("rfr", parent_only=True))) == 3
    assert client.create_study("rfr", run_number=-1).name == study.name


def test_set_terminated_all(client):
    assert client.set_terminated_all("rfr") is None


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


def test_load_instance(client):
    run = client.create_run("example")
    run.start("train")
    results = client.load_instance(run.id, "results", "test")
    assert "train" in results
    assert "val" in results
    assert "test" not in results
    model = client.load_instance(run.id, "model", "test")
    assert isinstance(model, torch.nn.Module)


def test_load_results(client):
    run = client.create_run("example")
    run.start("train")
    run.start("test")
    r = client.load_instance(run.id, "results", "test")
    c = client.load_results([run.id, run.id])
    assert len(c.val["index"]) == 2 * len(r.val["index"])
    assert len(c.test["index"]) == 2 * len(r.test["index"])


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


def test_run_tags(client):
    run0 = client.create_run("example", tags=dict(name="abc"))
    run1 = client.create_run("example", tags=dict(name="def"))
    run2 = client.create_run("example", tags=dict(name="abc"))
    assert client.get_run_id_by_tag("example", name="abc") == run2.id
    assert client.get_run_id_by_tag("example", name="def") == run1.id
    assert list(client.get_run_ids_by_tag("example", name="abc")) == [run2.id, run0.id]
