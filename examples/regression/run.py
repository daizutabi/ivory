from ivory.core.tracker import create_tracker
from ivory.core.experiment import create_experiment

import os
from ivory import utils
import mlflow
import tempfile

tracker = create_tracker("params.yaml")
run_id = tracker.get_run_id('9', 0)
client = mlflow.tracking.MlflowClient(tracker.tracking_uri)
with tempfile.TemporaryDirectory() as tmpdir:
    path = client.download_artifacts(run_id, 'params.yaml', tmpdir)
    params = utils.load_params(path)
experiment = create_experiment(params)
experiment.start()
experiment
run = experiment.create_run()

with tempfile.TemporaryDirectory() as tmpdir:
    path = client.download_artifacts(run_id, 'current', tmpdir)
    state_dict = run.load(path)

state_dict.keys()
run.load_state_dict(state_dict)


client.get_metric_history(run_id, 'best_score')
