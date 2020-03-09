import torch

import ivory.core.run


def dump(run):
    checkpoint = {x: run[x].state_dict() for x in run if hasattr(run[x], "state_dict")}
    checkpoint["config"] = run.config
    return checkpoint


def store(run, checkpoint):
    for x in checkpoint:
        if x == "config":
            run.config = checkpoint["config"]
        else:
            run[x].load_state_dict(checkpoint[x])


class Run(ivory.core.run.Run):
    def start(self):
        self.trainer.fit(self)

    def save(self, path):
        checkpoint = dump(self)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        store(self, checkpoint)
