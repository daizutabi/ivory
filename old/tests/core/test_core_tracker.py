import uuid

import torch

from ivory.core.tracker import get_valid_mode


def test_create_experiment(tracker):
    id1 = tracker.create_experiment(str(uuid.uuid4()))
    tracker.artifact_location = "./mlruns"
    id2 = tracker.create_experiment(str(uuid.uuid4()))
    assert int(id1) + 1 == int(id2)


def test_get_experiment_id(tracker):
    assert tracker.get_experiment_id("example")


def test_get_run_id(tracker, run):
    name, number = run.name.split("#")
    name = name.lower()
    number = int(number)
    assert tracker.get_run_id(run.experiment_id, name, number) == run.id


def test_get_valid_mode(tracker, client, run):
    for mode in ["best", "current", "test"]:
        assert get_valid_mode(tracker.client, run.id, mode) == mode

    run = client.create_run("ridge")
    run.start("train")
    # assert get_valid_mode(tracker.client, run.id, "test") == "current"


def test_load_run(tracker, run, experiment):
    run = tracker.load_run(run.id, "test")
    assert run.trainer.epoch != -1


def test_load_instance(tracker, run):
    results = tracker.load_instance(run.id, "results", "test")
    assert "test" in results

    model = tracker.load_instance(run.id, "model", "test")
    assert isinstance(model, torch.nn.Module)


def test_remove_delted_runs(tracker, experiment):
    run = experiment.create_run()
    run.start()
    tracker.client.delete_run(run.id)
    assert tracker.remove_deleted_runs(experiment.id) == 1
