import ivory.core.runner


class Runner(ivory.core.runner.Runner):
    def run(self, fold=0):
        train_loader, val_loader = self.dataloaders[fold]
        self.trainer.fit(train_loader, val_loader, self)
