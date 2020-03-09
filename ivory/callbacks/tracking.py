import mlflow

from ivory.callbacks import Callback


class Tracking(Callback):
    def on_fit_start(self, run):
        self.experiment_id = mlflow.get_experiment_by_name(run.experiment.name)
        if self.experiment_id is None:
            self.experiment_id = mlflow.create_experiment(run.experiment.name)
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=run.name) as r:
            self.run_id = r.info.run_id
            mlflow.log_params(run.params)

    def on_epoch_end(self, run):
        metrics = run.metrics
        with mlflow.start_run(self.run_id):
            mlflow.log_metrics(dict(metrics.current_recode), metrics.current_epoch)

    def on_fit_end(self, run):
        pass
        # metrics = run.metrics
        # with mlflow.start_run(self.run_id):
        #     mlflow.log_metrics(dict(metrics.current_recode), metrics.current_epoch)
