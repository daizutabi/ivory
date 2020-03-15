import numpy as np

from ivory.core.state import State


class Metrics(State):
    def __init__(self):
        self.epoch = -1
        self.record = {}
        self.history = {}

    def __repr__(self):
        class_name = self.__class__.__name__
        s = f"{class_name}(num_metrics={len(self.record)}, "
        s += f"num_records={len(self.history)})"
        return s

    def __str__(self):
        metrics = []
        for key in self.record:
            metrics.append(f"{key}={self.record[key]:.2e}")
        return " ".join(metrics)

    def reset(self):
        self.train_batch_loss = []
        self.val_batch_loss = []
        self.val_batch_index = []
        self.val_batch_output = []
        self.val_batch_target = []

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
        index, output, target, batch_loss = self.val_evaluate(index, output, target)
        self.val_batch_index.append(index)
        self.val_batch_output.append(output)
        self.val_batch_target.append(target)
        self.val_batch_loss.append(batch_loss)

    def val_evaluate(self, index, output, target):
        """Returns a result for a validation step.

        Args:
            index, output, target: batch data.

        Returns:
            tuple:
                (0-2) index, output, target (np.ndarray): numpay batch data.
                (3) batch_loss (float): a loss value
        """
        raise NotImplementedError

    def on_epoch_end(self, run):
        self.data = self.data_dict()
        train_epoch_loss = np.mean(self.train_batch_loss)
        val_epoch_loss = np.mean(self.val_batch_loss)
        self.record = {"loss": train_epoch_loss, "val_loss": val_epoch_loss}
        self.record.update(self.record_dict(run))
        self.history[self.epoch] = self.record
        self.reset()

    def data_dict(self):
        """Create data from validation data."""
        index = np.hstack(self.val_batch_index)
        output = np.vstack(self.val_batch_output)
        target = np.vstack(self.val_batch_target)
        return dict(index=index, output=output, target=target)

    def record_dict(self, run):
        """Returns an extra custom metrics dictionay."""
        return {}
