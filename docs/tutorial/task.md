# Multiple Runs

{{ ## cache:clear }}

## Task

Ivory implements a special run class `Task` that controls multiple nested runs. `Task` is useful for parameter search or cross validation.

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
    shutil.rmtree('examples/mlruns')
```

```python
import ivory

client = ivory.create_client("examples")  # Set the working directory
task = client.create_task('torch')  # Or, experiment.create_task()
task
```

The `Task` class has two functions to generate multiple runs: `Task.prodcut()` and `Task.chain()`. These two function have the same functionality as [`itertools`](https://docs.python.org/3/library/itertools.html) of Python starndard library.

### Product

The `Task.prodcut()` makes an iterator that returns runs from cartesian product of input parameters.

```python
task = client.create_task('torch')
# verbose=0: No progress bar.
runs = task.product(fold=range(2), factor=[0.5, 0.7], verbose=0)
runs
```

```python
for run in runs:
    pass  # Do somthing, for example, run.start()
```

You can specify other parameters that don't change during iteration.

```python
task = client.create_task('torch')
runs = task.product(fold=range(2), factor=[0.5, 0.7], lr=1e-4, verbose=0)
for run in runs:
    pass  # Do somthing, for example, run.start()
```


### Chain

The `Task.chain()` makes an iterator that returns runs from the first input paramter until it is exhausted, then proceeds to the next parameter, until all of the parameters are exhausted. Other parameters have default values if they don't be specified by additional key-value pairs.

```python
task = client.create_task('torch')
runs = task.chain(
    fold=range(2),
    factor=[0.5, 0.7],
    lr=[1e-4, 1e-3],
    batch_size=32,
    use_best_param=False,
    verbose=0)
runs
```

```python
for run in runs:
    pass  # Do somthing, for example, run.start()
```

The `use_best_param` keyword argument is useful for dynamic updating of parameters. If `True` (default), the parameter that got the best score is used during the following iterations.

```python
task = client.create_task('torch')
runs = task.chain(
    fold=range(2),
    factor=[0.5, 0.7],
    lr=[1e-4, 1e-3],
    use_best_param=True,
    verbose=0)
for run in runs:
    pass  # Do somthing, for example, run.start()
    # We do nothing, so the first values are used.
```

## Range

Ivory provides the `ivory.utils.range.Range` class for parameter ranging. This
class can be used as the standard `range`, but more flexible, especially for the float type.

```python
from ivory.utils.range import Range

list(Range(6))  # The stop value is included.
```

```python
list(Range(3, 6))  # Start and stop.
```

```python
list(Range(3, 10, 2))  # Step size.
```

```python
list(Range(3, 10, num=4))  # Sampling size.
```


```python
list(Range(0.0, 1.0, 0.25))  # float type.
```

```python
list(Range(0.0, 1.0, num=5))  # Sampling size
```

```python
list(Range(1e-3, 1e2, num=6, log=True))  # Log scale
```

A `Range` instance can be created from a string.

```python
list(Range('3-7'))  # <start>-<stop>
```

```python
list(Range('3-7-2')) # <start>-<stop>-<step>
```

```python
list(Range('0.0-1.0:5')) # <start>-<stop>:<num>
```

```python
list(Range('1e-3_1e2:6.log'))  # '_' instead of '-', log scale
```
