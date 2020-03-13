import pytest
import ivory


def test_experiment(path):
    experiment = ivory.create_experiment(path)
    ivory.active_experiment = None
    experiment.start()
    assert experiment is ivory.active_experiment
    assert experiment.name.startswith('2')

    assert all(experiment.default["data"] == [1, 2])
    assert "shared=['data']" in repr(experiment)

    run = experiment.create_run()
    assert run.name == "#1"

    run = ivory.create_run(experiment=experiment)
    assert run.name == "#2"

    run = ivory.create_run(experiment=experiment)
    assert run.name == "#3"

    run = ivory.create_run({"data2.object": [10, 20]})
    assert run.experiment is ivory.active_experiment
    assert all(run.data2 == [10, 20])

    ivory.active_experiment = None
    with pytest.raises(ValueError):
        ivory.create_run()
