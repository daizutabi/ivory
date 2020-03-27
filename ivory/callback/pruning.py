import numpy as np
import optuna
from optuna.trial import Trial


class Pruning:
    """Callback to prune unpromising trials.

    Args:
        trial:
            `optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        metric:
            An evaluation metric for pruning, e.g., `val_loss`
    """

    def __init__(self, trial: Trial, metric: str):
        self.trial = trial
        self.metric = metric

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(trial={self.trial}, metric='{self.metric}')"

    def on_epoch_end(self, run):
        score = run.metrics[self.metric]
        if np.isnan(score):
            return
        epoch = run.metrics.epoch
        self.trial.report(score, step=epoch)
        if self.trial.should_prune():
            message = f"Trial was pruned at epoch {epoch}."
            raise optuna.exceptions.TrialPruned(message)
