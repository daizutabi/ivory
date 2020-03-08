from dataclasses import dataclass

from ivory.core.callback import Callback


def dump_checkpoint(run):
    checkpoint = {x: run[x].state_dict() for x in run if hasattr(run[x], "state_dict")}
    checkpoint["config"] = run.config
    return checkpoint


@dataclass
class Checkpoint(Callback):
    def on_epoch_end(self, run):
        checkpoint = dump_checkpoint(run)
