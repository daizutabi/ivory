# Multiple Runs

{{ ## cache:clear }}

## Task

Ivory implements a special run type called **Task** which controls multiple nested runs. A task is useful for parameter search or cross validation.

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

The `Task` class has two methods to generate multiple runs: `prodcut()` and `chain()`. These two methods have the same functionality as [`itertools`](https://docs.python.org/3/library/itertools.html) of Python starndard library.

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

You can specify other parameters which don't change during iteration.

```python
task = client.create_task('torch')
runs = task.product(fold=range(2), factor=[0.5, 0.7], lr=1e-4, verbose=0)
for run in runs:
  pass  # Do somthing, for example, run.start()
```


### Chain

The `Task.chain()` maks an iterator that returns runs from the first input paramter until it is exhausted, then proceeds to the next parameter, until all of the parameters are exhausted. Other parameters have default values if they don't be specified by additional key-value pairs.

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

The `use_best_param` keyword argument is useful for dynamic updating of parameters. If `True` (default), the parameter which got the best score is used during the following iterations.

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


## Tracking

If the `Client` instace has a `Tracker` instance, the multiple runs created by the tasks can be tracked. The `client.search_parent_run_ids()` method makes an iterator that returns RunIDs of runs that have nested runs. In this case, parent runs are some tasks we made above.


```python
# A helper function
def print_run_info(run_ids):
    for run_id in run_ids:
        print(run_id[:5], client.get_run_name(run_id))

run_ids = client.search_parent_run_ids('torch')
print_run_info(run_ids)
```

!!! note
    `task#0` that we made first hasn't yielded any runs yet, so that the task has not been a parent run.


The `client.get_run_ids()` makes an iterator that returns RunIDs of runs you select by run names.

```python
run_ids = client.get_run_ids('torch', task=range(2,4))
print_run_info(run_ids)
```

The `client.get_nested_run_ids()` makes an iterator that returns RunIDs of runs that have a parent you select by run names.

```python
run_ids = client.get_nested_run_ids('torch', task=range(3, 5))
print_run_info(run_ids)
```

On the other hand, the `client.get_parent_run_id()` returns a RunID of a run that is refered by a nested run.

```python
run_id = client.get_parent_run_id('torch', run=14)
print_run_info([run_id])
```

## Range

Ivory provides the `ivory.utils.range.Range` class for parameter setting. This
class can be used as the standard `range`, but more flexible, expecially for float type.

```python
from ivory.utils.range import Range

# Normal usage
list(Range(3, 6))  # The stop value is included.
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
