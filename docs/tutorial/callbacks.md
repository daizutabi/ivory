# Callbacks

## Base System

Ivory implements a simple but powerful callback system.

Here is the list of callback functions:

{{ # cache:clear }}

```python
import ivory.core.base

ivory.core.base.Callback.METHODS
```

Any class that defines these functions can be a callback instance.

```python
class SimpleCallback:  # No base class is needed.
    # You don't have to define all of the callback functions
    def  on_fit_begin(self, run):  # Must have a `run` argument.
        print(f'on_fit_begin is called from id={id(run)}')
        # Do something with `run`.
```

To invoke callback functions, create a `CallbackCaller` instance.

```python
caller = ivory.core.base.CallbackCaller(simple=SimpleCallback())
caller
```

The number of registerd objects is 1.


```python
list(caller)
```

Then call the `CallbackCaller.create_callbacks()` method to build a callback network.

```python
caller.create_callbacks()
caller
```

The number of objects increased up to 13.

```python
list(caller)
```

Callback functions are added to the caller instance. Inspect each callback funtion.

```python
caller.on_init_begin
```

This is an empty callback because the caller has no objects that define the `on_init_begin()` function. On the other hand,


```python
caller.on_fit_begin
```

The `simple` object is registerd as a receiver for the `on_fit_begin()` function. We can call this.

```python
caller.on_fit_begin()
```

```python
id(caller)
```

This caller-receiver network among arbitrary object collection builds a complex machine learning workflow.

The `Run` class is a subclass of the `CallbackCaller` class and performs more library-specific process. We uses this `Run` class below.

## Example Preparation

To work with the callbacks, we create data and model set that we have used. For more details about the following code, see [Creating Instance](../instance) section.

```python
import yaml

# A helper function.
def create(doc, name, **kwargs):
    params = yaml.safe_load(doc)
    return create_instance(params, name, **kwargs)

doc = """
library: torch
dataloaders:
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
  fold: 0
  batch_size: 4
model:
  class: rectangle.torch.Model
  hidden_sizes: [3, 4, 5]
"""
dataloaders = create(doc, 'dataloaders')
model = create(doc, 'model')
```


## Results

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
    dataloaders=dataloaders,
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

Let's play with the `Results` callback. The `Results.step()` method records the current index, output, and target.

```python
# For simplicity, just one epoch with some batches.
run.on_train_begin()
data_iter = iter(run.dataloaders.train)
for _ in range(3):
    index, input, target = next(data_iter)
    output = model(input)
    run.results.step(index, output, target)
    # Do something for example parameter update or early stopping.
run.on_train_end()
run.on_val_begin()  # Can call even if there is no callback.
data_iter = iter(run.dataloaders.val)
for _ in range(2):
    index, input, target = next(data_iter)
    output = run.model(input)
    run.results.step(index, output, target)
run.on_val_end()
run.on_epoch_end()

results
```

We performed a train and validation loop so that the `Results` instance has these data, but doesn't have test data. We can get data by nested dot-notation.

```python
results.train
```

```python
results.train.index  # Shuffled. The length is batch_size (4) x 3.
```

```python
results.val.index  # Not shuffled. The length is batch_size (4) x 2.
```

```python
results.val.output  # Actually, no learning.
```

```python
results.val.target
```

## Other Callback

There are several callback such as `Metrics`, `Monitor`, *etc*. We will learn about them in next ['Training a Model'](../trainer) section.
