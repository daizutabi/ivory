import datetime
import os
import tempfile

import mlflow

import ivory
from ivory import utils
from ivory.core.experiment import create_experiment
from ivory.core.tracker import create_tracker
from ivory.utils import load_params

environment = ivory.create_environment("params.yaml")
experiment = environment.create_experiment("params.yaml")
experiment.objective.set_params("params.yaml")
objective = experiment.create_objective()
study = experiment.create_study()


study.optimize(objective, n_trials=4)

experiment.objective.create_params(1)

run = experiment.create_run("params.yaml")
run


run.dataloader.val_dataloader.dataset.index
run.tracking
run.start()
run


run = ivory.create_run("params.yaml")
run.connect(tracker, experiment.id)
run.dataloader(experiment.data)
run.dataloader.train_dataloader.dataset[0]
run.create_callbacks()
run.trainer.fit(run)
experiment.objective



load_params("params.yaml")["run"]


experiment.objects

datetime.datetime.now().strftime("E%Y%m%d%H%M%S")



tracker = create_tracker("params.yaml")
run_id = tracker.get_run_id("9", 0)
client = mlflow.tracking.MlflowClient(tracker.tracking_uri)
with tempfile.TemporaryDirectory() as tmpdir:
    path = client.download_artifacts(run_id, "params.yaml", tmpdir)
    params = utils.load_params(path)
experiment = create_experiment(params)
experiment.start()
experiment
run = experiment.create_run()

with tempfile.TemporaryDirectory() as tmpdir:
    path = client.download_artifacts(run_id, "current", tmpdir)
    state_dict = run.load(path)

state_dict.keys()
run.load_state_dict(state_dict)


client.get_metric_history(run_id, "best_score")
