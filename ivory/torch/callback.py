from dataclasses import dataclass

from ivory.core.callback import Callback


@dataclass
class Checkpoint(Callback):
    def on_epoch_end(self, run):
        pass
        # checkpoint = run.dump()
