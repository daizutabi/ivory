import ivory


def test_create_experiment(params_path):
    experiment = ivory.create_experiment(params_path)
    assert experiment.name == ""


def test_create_experiment_from_envrionment(experiment):
    assert experiment.name == "Default"
    assert experiment.tuner
    assert experiment.tracker


def test_create_objective(experiment, run):
    objective = experiment.create_objective(run.params)
    assert callable(objective)


def test_create_study(experiment, run):
    study = experiment.create_study()
    assert "experiment_id" in study.user_attrs
