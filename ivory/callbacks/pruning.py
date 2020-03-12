import optuna

from ivory.callbacks import Callback
from optuna.trial import Trial


class Pruning(Callback):
    """Callback to prune unpromising trials.

    Args:
        trial:
            `optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., `val_loss`
    """

    def __init__(self, trial: Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, run):
        current_score = run.metrics.current_record[self.monitor]
        if current_score is None:
            return
        epoch = run.metrics.current_epoch
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = f"Trial was pruned at epoch {epoch}."
            raise optuna.exceptions.TrialPruned(message)
