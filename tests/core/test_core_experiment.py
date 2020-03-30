def test_experiment(experiment):
    assert experiment.name == 'example'
    assert experiment.tuner
    assert experiment.tracker
