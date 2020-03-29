import pytest


def test_tuner(experiment):
    tuner = experiment.tuner
    study = tuner.create_study("abc", mode="min")
    assert "MINIMIZE" in str(study.direction)
    study = tuner.create_study("abc", mode="max")
    assert "MAXIMIZE" in str(study.direction)

    with pytest.raises(ValueError):
        tuner.create_study("def", mode="mean")
