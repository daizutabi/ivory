import tempfile


import ivory
from ivory import utils

tracker = ivory.create_instance("params.yaml", 'environment.tracker')

environment = ivory.create_environment("params.yaml")
experiment = environment.create_experiment("params.yaml")
objective = experiment.create_objective("params.yaml")
study = experiment.create_study()
study.optimize(objective, n_trials=4)
experiment.id
tracker = environment.tracker
run_infos = tracker.list_run_infos("31")
run_id = run_infos[0].run_id


client = tracker.client
with tempfile.TemporaryDirectory() as tmpdir:
    path = client.download_artifacts(run_id, "params.yaml", tmpdir)
    params = utils.load_params(path)
    run = experiment.create_run(params)
    path = client.download_artifacts(run_id, "current", tmpdir)
    state_dict = run.load(path)
    run.load_state_dict(state_dict)

experiment.name

experiment = tracker.client.get_experiment_by_name(experiment.name)
run.start()
