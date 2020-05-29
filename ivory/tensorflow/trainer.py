import ivory.core.estimator
from ivory.core import instance
from ivory.core.exceptions import EarlyStopped, Pruned
from ivory.core.run import Run
from ivory.core.trainer import message
from ivory.tensorflow.callback import Callback
from ivory.utils.tqdm import tqdm


class Trainer(ivory.core.estimator.Estimator):
    def __init__(self, epoch=-1, epochs=1, batch_size=32, verbose=1, **kwargs):
        self.epoch = epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.kwargs = kwargs

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}()"

    def on_init_begin(self, run):
        if not run.model._is_compiled:
            params, self.kwargs = instance.filter_params(run.model.fit, **self.kwargs)
            if run.optimizer:
                params.update(optimizer=run.optimizer)
            run.model.compile(**params)

    def train(self, run: Run):
        run.on_fit_begin()
        callbacks = [Callback(run)]
        if run.scheduler:
            callbacks.insert(0, run.scheduler)
        if "callbacks" in self.kwargs:
            callbacks = self.kwargs.pop("callbacks") + callbacks
        train_dataset = run.datasets.train[:][1:]
        val_dataset = run.datasets.val[:][1:]
        try:
            run.model.fit(
                *train_dataset,
                validation_data=val_dataset,
                batch_size=self.batch_size,
                initial_epoch=self.epoch + 1,
                epochs=self.epoch + self.epochs + 1,
                callbacks=callbacks,
                verbose=0,
                **self.kwargs,
            )
        except (EarlyStopped, Pruned):
            pass
        finally:
            run.on_fit_end()

    def test(self, run: Run):
        run.on_test_begin()
        index, input, *target = run.datasets.test[:]
        output = run.model.predict(input, callbacks=[Callback(run)])
        if run.results:
            run.results.step(index, output, *target)
        run.on_test_end()

    def log(self, run: Run, early_stopped=False, pruned=False):
        msg = message(run, early_stopped, pruned)
        if msg:
            tqdm.write(msg)

    def on_fit_begin(self, run: Run):
        if self.verbose == 1:
            self.epoch_bar = tqdm(total=self.epochs, desc="Epoch", leave=False)

    def on_epoch_end(self, run: Run):
        if self.verbose == 1:
            self.epoch_bar.update(1)

    def on_fit_end(self, run: Run):
        if self.verbose == 1:
            self.epoch_bar.close()

    def on_loop_begin(self, run: Run, mode: str):
        if self.verbose == 1:
            dataset = run.datasets[mode]
            total = len(dataset) // self.batch_size
            mode = "%-5s" % (mode[0].upper() + mode[1:])
            self.batch_bar = tqdm(total=total, desc=mode, leave=False)

    def on_batch_end(self):
        if self.verbose == 1:
            self.batch_bar.update(1)

    def on_train_begin(self, run: Run):
        self.on_loop_begin(run, "train")

    def on_train_end(self, run: Run):
        if self.verbose == 1:
            self.batch_bar.close()

    def on_val_begin(self, run: Run):
        self.on_loop_begin(run, "val")

    def on_val_end(self, run: Run):
        if self.verbose == 1:
            self.batch_bar.close()

    def on_test_begin(self, run: Run):
        self.on_loop_begin(run, "test")

    def on_test_end(self, run: Run):
        if self.verbose == 1:
            self.batch_bar.close()

    def state_dict(self):
        state_dict = super().state_dict()
        if "epoch_bar" in state_dict:
            del state_dict["epoch_bar"]
        if "batch_bar" in state_dict:
            del state_dict["batch_bar"]
        return state_dict
