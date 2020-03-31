import uuid

import torch

from ivory.core.tracker import get_modes, get_valid_mode


def test_create_experiment(experiment):
    tracker = experiment.tracker
    id1 = tracker.create_experiment(str(uuid.uuid4()))
    tracker.artifact_location = "./mlruns"
    id2 = tracker.create_experiment(str(uuid.uuid4()))
    assert int(id1) + 1 == int(id2)


def test_get_modes(tracker, run):
    modes = get_modes(tracker.client, run.id)
    for mode in ["best", "current", "test"]:
        assert mode in modes


def test_get_valid_mode(tracker, run):
    for mode in ["best", "current", "test"]:
        mode_ = get_valid_mode(tracker.client, run.id, mode)
        assert mode_ == mode


def test_load_run(tracker, run, client):
    run = tracker.load_run(run.id, "test", client.create_run)
    assert run.trainer.epoch != -1


def test_load_instance(tracker, run, client):
    results = tracker.load_instance(
        run.id, "results", "test", client.create_run, client.create_instance
    )
    assert "test" in results

    model = tracker.load_instance(
        run.id, "model", "test", client.create_run, client.create_instance
    )
    assert isinstance(model, torch.nn.Module)
