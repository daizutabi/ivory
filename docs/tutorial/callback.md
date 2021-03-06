# Callback System

{{ ## cache:clear }}

## Basics

Ivory implements a simple but powerful callback system.

Here is the list of callback functions in the order of invocation:

```python
import ivory.core.base

ivory.core.base.Callback.METHODS
```

Any class that defines these functions can be a callback.

```python
class SimpleCallback:  # No base class is needed.
    # You don't have to define all of the callback functions
    def on_fit_begin(self, run):  # Must have an only `run` argument.
        print(f'on_fit_begin is called from id={id(run)}')
        # Do something with `run`.
```

To invoke callback functions, create a `CallbackCaller` instance.

```python
caller = ivory.core.base.CallbackCaller(simple=SimpleCallback())
caller
```

The number of registered instances is 1.

```python
list(caller)
```

Then call `CallbackCaller.create_callbacks()` to build a callback network.

```python
caller.create_callbacks()
caller
```

The number of instances increased up to 13.

```python
list(caller)
```

Callback functions are added to the caller instance. Let's inspect some callback functions.

```python
caller.on_init_begin
```

This is an empty callback because the caller has no instances that define the `on_init_begin()`. On the other hand,


```python
caller.on_fit_begin
```

The `simple` instance is registered as a receiver of the `on_fit_begin()`. We can call this.

```python
caller.on_fit_begin()
```

```python
id(caller)
```

This caller-receiver network among arbitrary instance collection builds a complex machine learning workflow.

`Run` class is a subclass of the `CallbackCaller` and performs more library-specific process. We uses `Run` below.

## Example Callback: Results

To work with the `Results` callback, we create a set of data and a model. For more details about the following code, see [Creating Instance](../instance) section.

```python
import yaml
from ivory.core.instance import create_instance

# A helper function.
def create(doc, name, **kwargs):
    params = yaml.safe_load(doc)
    return create_instance(params, name, **kwargs)

doc = """
library: torch
datasets:
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
  fold: 0
model:
  class: rectangle.torch.Model
  hidden_sizes: [3, 4, 5]
"""
datasets = create(doc, 'datasets')
model = create(doc, 'model')
```

The `Results` callback stores index, output, and target data. To save memory, a `Results` instance ignores input data.

```python
# import ivory.callbacks.results  # For Scikit-learn or TensorFlow.
import ivory.torch.results

results = ivory.torch.results.Results()
results
```

```python
import ivory.core.run

run = ivory.core.run.Run(
    datasets=datasets,
    model=model,
    results=results
)
run.create_callbacks()
run
```

```python
# A helper function
def print_callbacks(obj):
    for func in ivory.core.base.Callback.METHODS:
        if hasattr(obj, func) and callable(getattr(obj, func)):
            print(func)

print_callbacks(results)  
```

Let's play with the `Results` callback. `Results.step()` records the current index, output, and target.

```python
import torch

# For simplicity, just one epoch with some batches.
run.on_train_begin()
dataset = run.datasets.train
for k in range(3):
    index, input, target = dataset[4 * k : 4 * (k + 1)]
    input, target = torch.tensor(input), torch.tensor(target)
    output = run.model(input)
    run.results.step(index, output, target)
    # Do something for example parameter update or early stopping.
run.on_train_end()
run.on_val_begin()  # Can call even if there is no callback.
dataset = run.datasets.val
for k in range(2):
    index, input, target = dataset[4 * k : 4 * (k + 1)]
    input, target = torch.tensor(input), torch.tensor(target)
    output = run.model(input)
    run.results.step(index, output, target)
run.on_val_end()
run.on_epoch_end()

results
```

We performed a train and validation loop so that the `Results` instance has these data, but doesn't have test data. 

```python
results.train
```

```python
results.train.index  # The length is 4 x 3.
```

```python
results.val.index  # The length is 4 x 2.
```

```python
results.val.output
```

```python
results.val.target
```

## Other Callback

There are several callback such as `Metrics`, `Monitor`, *etc.* We will learn about them in next [Training a Model](../training) tutorial.
