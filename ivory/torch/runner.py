import ivory


class Runner(ivory.Runner):
    def run(self, fold=0):
        train_loader, val_loader = self.cfg.dataloaders[fold]
        self.cfg.trainer.fit(train_loader, val_loader, self.cfg)
