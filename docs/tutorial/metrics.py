# # Metrics

# Ivory has a `ivory.callback.Metrics` that doesn't depend on any specific library such
# as PyTorch or scikit-learn, *etc.* As the module name shows, a  `Metrics` is a
# callback called from a `Trainer`, which is created by a `Run`.

# So at this stage, we cannot get an instance of `Metrics`. Instead, let's check the
# `ivory.torch.Metrics` that is a metrics class for PyTorch. A instance of this class
# will be called from `ivory.torch.Trainer`.

# #File {%=/ivory/torch/metrics.py%} ivory/torch/metrics.py

# There four instance methods. `train_evaluate` and `val_evaluate` are functions that
# called from train and validation loops. `evalute` is a customizable function for
# metrics at a step. You can overwrite this function, for example:

from typing import Dict

import torch

import ivory.torch


class MyMetrics(ivory.torch.Metrics):
    def evaluate(self, loss, output, target) -> Dict[str, float]:
        mse = torch.mean((output - target) ** 2).item()
        return {"loss": loss.item(), "mse": mse}  # Here, loss == mse


# The last method `on_current_record` is called from `on_epoch_end` callback function.
# This gives you a chance to modify a current metrics for epoch (we call it a `record`.)
# In the above case, we add the learning rate of our optimizer in order to monitor it
# during a traing loop. A callback function of Ivoy takes one argument `run` which is
# corresponding to the current run. You can access all of the run information via this
# `run` instance.
