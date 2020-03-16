import optuna

from ivory.core.experiment import create_experiment

experiment = create_experiment('params.yaml')
experiment.start()
run = experiment.create_run()
run.start()
run
experiment = create_experiment(run.params)
experiment.start()
experiment
run = experiment.create_run()
run

run.params
run.params['experiment']['id']

run.params
experiment.objects
experiment
run.objects

from ivory.core.tracker import create_tracker

tracker = create_tracker("params.yaml")
import os

path = tracker.get_artifact_path('9', 0)
experiment = create_experiment(a)
experiment.start()
run = experiment.create_run()
run.start()
import mlflow

client = mlflow.tracking.MlflowClient(tracker.tracking_uri)
run_infos = client.list_run_infos('9')
run_info = run_infos[0]
run_id = run_info.run_id


a =client.download_artifacts(run_id, 'params.yaml', 'C:/Users/daizu/Desktop')
run = client.get_run(run_info.run_id)
path = run_info.artifact_uri[8:]
import os

os.listdir(path)

with tracker.download_artifacts(experiment_id, run_index):
