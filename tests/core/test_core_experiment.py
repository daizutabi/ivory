import ivory


def test_experiment(path):
    experiment = ivory.create_experiment(path)
    experiment.start()
    assert experiment.name.startswith("2")

    assert all(experiment.shared_objects["data"] == [1, 2])
    assert "shared=['data']" in repr(experiment)

    run = experiment.create_run()
    assert run.name == "#1"

    run = experiment.create_run()
    assert run.name == "#2"

    run = experiment.create_run({"data2.object": [10, 20]})
    assert all(run.data2 == [10, 20])


def test_crate_run(path):
    experiment = ivory.create_experiment(path)
    experiment.create_run()
    assert experiment.name.startswith("2")
