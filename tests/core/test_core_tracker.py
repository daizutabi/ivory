import uuid


def test_create_experiment(experiment):
    tracker = experiment.tracker
    id1 = tracker.create_experiment(str(uuid.uuid4()))
    tracker.artifact_location = "./mlruns"
    id2 = tracker.create_experiment(str(uuid.uuid4()))
    assert int(id1) + 1 == int(id2)
