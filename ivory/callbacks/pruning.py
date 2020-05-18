"""Pruning class to prune unpromising trials."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna
from optuna.trial import Trial

from ivory.core.exceptions import Pruned
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

    trial: Optional[Trial] = None
    metric: str = ""

    def on_epoch_end(self, run: Run):
        if self.trial is not None:
            score = run.metrics[self.metric]
            if np.isnan(score):
                return
            epoch = run.metrics.epoch
            self.trial.report(score, step=epoch)
            if self.trial.should_prune():
                message = f"Trial was pruned at epoch {epoch}."
                raise optuna.exceptions.TrialPruned(message)

        if run.tracking:
            status = run.tracking.client.get_run(run.id).info.status
            if status == "KILLED":
                raise Pruned
