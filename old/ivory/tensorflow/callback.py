import tensorflow.keras.callbacks
from optuna.exceptions import TrialPruned

from ivory.core.exceptions import EarlyStopped, Pruned
from ivory.core.run import Run


class Callback(tensorflow.keras.callbacks.Callback):
    """Callback class used to connect Ivory and Tensorflow's original `Callback`.

    Args:
        run: A `Run` instance connected to this `Callback`.
    """

    def __init__(self, run: Run):
        self.run = run
        self.trainer = run.trainer
        self.predict = run.model.predict

    def on_epoch_begin(self, epoch, logs=None):
        """Calls `run.on_epoch_begin()` and `run.on_train_begin()`."""
        self.trainer.epoch = epoch
        self.run.on_epoch_begin()
        self.run.on_train_begin()

    def on_test_begin(self, logs=None):
        """Calls `run.on_train_end()` and `run.on_val_begin()`."""
        self.trainer.step(self.run, "train", training=False, predict=self.predict)
        self.run.on_train_end()
        self.run.on_val_begin()

    def on_test_end(self, logs=None):
        """Calls `run.on_val_end()`."""
        self.trainer.step(self.run, "val", predict=self.predict)
        self.run.on_val_end()

    def on_epoch_end(self, epoch, logs=None):
        """Calls `run.on_epoch_end()`."""
        if self.run.metrics and logs:
            for key, value in logs.items():
                key = key.replace("accuracy", "acc")
                self.run.metrics[key] = value
        early_stopped = pruned = None
        try:
            self.run.on_epoch_end()
        except EarlyStopped as e:
            early_stopped = e
        except Pruned as e:
            pruned = e
        except TrialPruned as e:
            pruned = e
        finally:
            self.trainer.log(self.run, early_stopped, pruned)
        if pruned:
            raise pruned

    def on_train_batch_end(self, batch, logs=None):
        """Call `trainer.on_batch_end` to update a progress bar."""
        self.trainer.on_batch_end()

    def on_test_batch_end(self, batch, logs=None):
        """Call `trainer.on_batch_end` to update a progress bar."""
        self.trainer.on_batch_end()

    def on_predict_batch_end(self, batch, logs=None):
        """Call `trainer.on_batch_end` to update a progress bar."""
        self.trainer.on_batch_end()
