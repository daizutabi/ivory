def test_experiment(experiment):
    assert experiment.name == 'Default'
    assert experiment.tuner
    assert experiment.tracker
