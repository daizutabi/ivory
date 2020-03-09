from dataclasses import dataclass

from ivory.callbacks import Callback


@dataclass
class Checkpoint(Callback):
    def on_epoch_end(self, run):
        pass
        # checkpoint = run.dump()
