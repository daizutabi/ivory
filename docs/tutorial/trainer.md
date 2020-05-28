# Training a Model

{{ ## cache:clear }}

First, create data and model set. For more details about the following code, see [Creating Instance section](../instance).

```python
import yaml

params = yaml.safe_load("""
library: torch
run:
  dataloaders:
    data:
      class: rectangle.data.Data
      n_splits: 4
    dataset:
    batch_size: 10
    fold: 0
  model:
    class: rectangle.torch.Model
    hidden_sizes: [100, 100]
  optimizer:
    class: torch.optim.SGD
    params: $.model.parameters()
    lr: 0.001
  scheduler:
    class: torch.optim.lr_scheduler.ReduceLROnPlateau
    optimizer: $
    factor: 0.5
    patience: 4
  results:
  metrics:
  monitor:
    metric: val_loss
  early_stopping:
    patience: 10
  trainer:
    loss: torch.nn.functional.mse_loss
    epochs: 10
    verbose: 2
""")
params
```

!!! note
    Key-order in the `params` dictionary is meaningful, because the callback functions are called by this order. For example, `Monitor` uses the results of `Metrics` so that `Monitor` should appear later than `Metrics`.

The `ivory.core.instance.create_base_instance()` function is more useful to create a run from a dictionary than the `ivory.core.instance.create_instance()` function because it can create multiple objects by one step.

```python
import ivory.core.instance

run = ivory.core.instance.create_base_instance(params, 'run')
list(run)
```

## Callbacks

Check callbacks of the `Run` instance.

```python
import ivory.core.base

# A helper function
def print_callbacks(obj):
    for func in ivory.core.base.Callback.METHODS:
        if hasattr(obj, func) and callable(getattr(obj, func)):
            print('  ', func)

for name, obj in run.items():
    print(f'[{name}]')
    print_callbacks(obj)
```

### Metrics

The role of `Metrics` class is to record a set of metric for evaluation of model performance. The metirics are updated at each epoch end.


```python
run.metrics  # Now, metrics are empty.
```

### Monitor

The `Monitor` class is monitoring the most important metric to measure the model score or to determine the training logic (early stopping or pruning).

```python
run.monitor  # Monitoring `val_loss`.  Lower is better.
```

### EarlyStopping

The `EarlyStopping` class is to stop the training loop when a monitored metric has stopped improving.

```python
run.early_stopping  # Early stopping occurs when `wait` > `patience`.
```

### Trainer

The `Tainer` class controls the model training. This is a callback, but at the same time, invokes callback functions at each step of training, validation, and test loop.

```python
run.trainer  # Training hasn't started yet, so epoch = -1.
```


## Using a Trainer

A `Run` instance invokes its trainer by `Run.start()` method.

```python
run.start()  # create_callbacks() is called automatically.
```

You can update attributes of run's objects at any time.

```python
run.trainer.epochs = 5
run.start()
```

!!! note
    The `Run.start()` method doesn't reset the trainer's epoch.

## Callbacks after Training

After training, the callbacks changes their states.

```python
run.metrics  # Show metrics at current epoch.
```

```python
run.metrics.history.val_loss  # Metrics history.
```

```python
run.monitor  # Store the best score and its epoch.
```

```python
run.early_stopping  # Current `wait`.
```

```python
run.trainer  # Current epoch is 14 (0-indexed).
```
