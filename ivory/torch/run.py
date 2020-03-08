import ivory.core.run


class Run(ivory.core.run.Run):
    def start(self, fold=0):
        train_loader, val_loader = self.dataloaders[fold]
        self.trainer.fit(train_loader, val_loader, self)

        # if (
        #     isinstance(run.scheduler, ReduceLROnPlateau)
        #     and run.metrics.direction != "minimize"
        # ):
        #     raise ValueError("metrics direction should be 'minimize'")

    def dump(self):
        checkpoint = {
            x: self[x].state_dict() for x in self if hasattr(self[x], "state_dict")
        }
        checkpoint["config"] = self.config
        return checkpoint

    def on_epoch_end(self, run):
        pass
        # checkpoint = run.dump()
