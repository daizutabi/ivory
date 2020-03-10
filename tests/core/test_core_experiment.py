import pytest
import ivory


def test_experiment(path):
    experiment = ivory.create_experiment(path)
    ivory.active_experiment = None
    experiment.start()
    assert experiment is ivory.active_experiment
    assert experiment.name == "example"
    assert experiment.run_name == "abc"
    params = experiment.params({"experiment": {"run_name": "def"}})
    assert params["experiment"]["run_name"] == "def"
    params = experiment.params({"experiment.run_name": "ghi"})
    assert params["experiment"]["run_name"] == "ghi"

    experiment.set_default(["data"])
    assert all(experiment.default["data"] == [1, 2])

    run = experiment.create_run({"experiment.run_name": "new"})
    assert run.name == "new"

    run = ivory.create_run(experiment=experiment)
    assert run.name == "abc"

    run = ivory.create_run({"experiment.run_name": "jkl"}, experiment=experiment)
    assert run.name == "jkl"

    run = ivory.create_run()
    assert run.experiment is ivory.active_experiment

    ivory.active_experiment = None
    with pytest.raises(ValueError):
        ivory.create_run()
