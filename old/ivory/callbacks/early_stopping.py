"""Early stopping when a monitored metric has stopped improving."""
from dataclasses import dataclass, field

from ivory.core.exceptions import EarlyStopped
from ivory.core.run import Run
from ivory.core.state import State


@dataclass
class EarlyStopping(State):
    """Early stops a training loop when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement after which training will be
            stopped.

    Attributes:
        patience: Number of epochs with no improvement after which training will be
            stopped.
        wait: Number of continuous epochs with no imporovement.

    Raises:
        EarlyStopped: When ealry stopping occurs.
    """

    patience: int
    wait: int = field(default=0, init=False)

    def on_epoch_end(self, run: Run):
        if run.monitor.is_best:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                raise EarlyStopped
