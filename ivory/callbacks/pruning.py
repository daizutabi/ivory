"""Pruning class to prune unpromising trials."""
from dataclasses import dataclass

import numpy as np
import optuna
from optuna.trial import Trial

from ivory.core.run import Run


@dataclass
class Pruning:
    """Callback to prune unpromising trials.

    Args:
        trial:
            A `Trial` corresponding to the current evaluation of the
            objective function.
        metric:
            An evaluation metric for pruning, e.g., `val_loss`
    """

    trial: Trial
    metric: str

    def on_epoch_end(self, run: Run):
        score = run.metrics[self.metric]
        if np.isnan(score):
            return
        epoch = run.metrics.epoch
        self.trial.report(score, step=epoch)
        if self.trial.should_prune():
            message = f"Trial was pruned at epoch {epoch}."
            raise optuna.exceptions.TrialPruned(message)
