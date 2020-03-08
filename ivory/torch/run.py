import ivory.core.run


class Run(ivory.core.run.Run):
    def start(self, fold=0):
        train_loader, val_loader = self.dataloaders[fold]
        self.trainer.fit(train_loader, val_loader, self)

    def dump(self):
        checkpoint = {
            x: self[x].state_dict() for x in self if hasattr(self[x], "state_dict")
        }
        checkpoint["config"] = self.config
        return checkpoint
