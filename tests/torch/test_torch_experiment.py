import ivory
from ivory.torch import Run


def test_experiment(tracking_uri):
    experiment = ivory.create_experiment("tests/torch/params.yaml")
    assert experiment.experiment_id == ""
    assert experiment.name == "ready"
    assert experiment.run_class == "ivory.torch.Run"
    assert experiment.run_cls is Run

    experiment.start()
    assert "pytest-" in experiment.tracker.tracking_uri
    experiment.optimize()

    experiment = ivory.create_experiment(
        "tests/torch/params.yaml",
        {
            "experiment.tracker.tracking_uri": tracking_uri,
            "experiment.tracker.artifact_location": "~/.mlruns2",
        },
    )
    experiment.start()
    assert experiment.tracker.tracking_uri == tracking_uri
    assert "file" in experiment.tracker.artifact_location