import pytest


def test_tuner(environment):
    tuner = environment.tuner
    study = tuner.create_study("abc", mode="max")
    assert "MAXIMIZE" in str(study.direction)

    with pytest.raises(ValueError):
        tuner.create_study("def", mode="mean")
