import tensorflow.keras.callbacks


class Callback(tensorflow.keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run
        self.trainer = run.trainer

    def on_train_begin(self, logs=None):
        self.run.on_fit_begin()

    def on_epoch_begin(self, epoch, logs=None):
        self.trainer.epoch = epoch
        self.run.on_epoch_begin()
        self.run.on_train_begin()

    def on_test_begin(self, logs=None):
        self.trainer.step(self.run, "train", training=False)
        self.run.on_train_end()
        self.run.on_val_begin()

    def on_test_end(self, logs=None):
        self.trainer.step(self.run, "val")
        self.run.on_val_end()

    def on_epoch_end(self, epoch, logs=None):
        if self.run.metrics and logs:
            for key, value in logs.items():
                key = key.replace("accuracy", "acc")
                self.run.metrics[key] = value
        self.run.on_epoch_end()
        self.trainer.log(self.run)

    def on_train_end(self, logs=None):
        self.run.on_fit_end()

    def on_train_batch_end(self, batch, logs=None):
        self.trainer.on_batch_end()

    def on_test_batch_end(self, batch, logs=None):
        self.trainer.on_batch_end()

    def on_predict_batch_end(self, batch, logs=None):
        self.trainer.on_batch_end()
