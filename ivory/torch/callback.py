from dataclasses import dataclass

from ivory.core.callback import Callback


def dump_checkpoint(obj):
    checkpoint = {x: obj[x].state_dict() for x in obj if hasattr(obj[x], "state_dict")}
    checkpoint["config"] = obj.config
    return checkpoint


@dataclass
class Checkpoint(Callback):
    def on_epoch_end(self, obj):
        checkpoint = dump_checkpoint(obj)
