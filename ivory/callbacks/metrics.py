import numpy as np

from ivory.core.state import State


class Metrics(State):
    def __init__(self):
        self.epoch = -1
        self.record = {}
        self.history = {}

    def __repr__(self):
        class_name = self.__class__.__name__
        args = str(self).replace(" ", ", ")
        return f"{class_name}({args})"

    def __str__(self):
        metrics = []
        for key in self.record:
            metrics.append(f"{key}={self.record[key]:.4g}")
        return " ".join(metrics)

    def reset(self):
        self.train_batch_loss = []
        self.val_batch_loss = []
        self.batch_index = []
        self.batch_output = []
        self.batch_target = []

    def on_epoch_start(self, run):
        self.epoch = run.trainer.epoch
        self.reset()

    def train_step(self, index, output, target):
        loss, batch_loss = self.train_evaluate(index, output, target)
        self.train_batch_loss.append(batch_loss)
        return loss

    def train_evaluate(self, index, output, target):
        """Returns a result for a training step.

        Args:
            index, output, target: batch data.

        Returns:
            tuple:
                (0) loss (object): a loss object (ex. torch.Tensor).
                (1) batch_loss (float): a loss value
        """
        raise NotImplementedError

    def val_step(self, index, output, target):
        index, output, target, *batch_loss = self.val_evaluate(index, output, target)
        if batch_loss:
            self.val_batch_loss.append(batch_loss[0])
        self.batch_index.append(index)
        self.batch_output.append(output)
        self.batch_target.append(target)

    def val_evaluate(self, index, output, target):
        """Returns a result for a validation step.

        Args:
            index, output, target: batch data.

        Returns:
            tuple:
                (0-2) index, output, target (np.ndarray): numpay batch data.
                (3, Optional) batch_loss (float): a loss value
        """
        return index, output, target

    def on_val_end(self, run):
        self.data = self.data_dict()

    def on_epoch_end(self, run):
        train_epoch_loss = np.mean(self.train_batch_loss)
        val_epoch_loss = np.mean(self.val_batch_loss)
        self.record = {"loss": train_epoch_loss, "val_loss": val_epoch_loss}
        self.record.update(self.record_dict(run))
        self.update_history()
        self.reset()

    def on_test_start(self, run):
        self.reset()

    def test_step(self, index, output):
        index, output = self.test_evaluate(index, output)
        self.batch_index.append(index)
        self.batch_output.append(output)

    def test_evaluate(self, index, output):
        """Returns a result for a validation step.

        Args:
            index, output, target: batch data.

        Returns:
            tuple:
                (0-2) index, output, target (np.ndarray): numpay batch data.
                (3, Optional) batch_loss (float): a loss value
        """
        return index, output

    def on_test_end(self, run):
        self.pred = self.data_dict()

    def update_history(self):
        for metric, value in self.record.items():
            if metric not in self.history:
                self.history[metric] = {self.epoch: value}
            else:
                self.history[metric][self.epoch] = value

    def data_dict(self):
        """Create data from validation/test data."""
        if self.batch_index[0].ndim == 1:
            index = np.hstack(self.batch_index)
        else:
            index = np.vstack(self.batch_index)
        if self.batch_output[0].ndim == 1:
            output = np.hstack(self.batch_output)
        else:
            output = np.vstack(self.batch_output)

        if len(self.batch_target) == 0:
            return dict(index=index, output=output)

        if self.batch_target[0].ndim == 1:
            target = np.hstack(self.batch_target)
        else:
            target = np.vstack(self.batch_target)
        return dict(index=index, output=output, target=target)

    def record_dict(self, run):
        """Returns an extra custom metrics dictionay."""
        return {}
