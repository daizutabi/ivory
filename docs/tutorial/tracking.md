# Tracking Runs with Ivory

{{ ## cache:clear }}

First create several runs for demonstration.

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
    shutil.rmtree('examples/mlruns')
```

```python
import ivory

client = ivory.create_client("examples")
run = client.create_run('torch', fold=2)
run.start()
```

```python
run = client.create_run('torch', fold=3)
run.start('both')
```

```python
task = client.create_task('torch')
runs = task.product(fold=range(3), verbose=0)
for run in runs:
    pass
    # Do something
```

```python
task = client.create_task('torch')
runs = task.product(n_splits=[3, 4], verbose=0)
for run in runs:
    pass
    # Do something
```

```python
task = client.create_task('torch')
runs = task.chain(lr=[1e-4, 1e-3], batch_size=[16, 32], verbose=0)
for run in runs:
    pass
    # Do something
```

## Tracking Interface

### Search functions

`Client.search_run_ids()` makes an iterator that returns Run IDs of runs.

```python
# A helper function
def print_run_info(run_ids):
    for run_id in run_ids:
        print(run_id[:5], client.get_run_name(run_id))

run_ids = client.search_run_ids('torch')
print_run_info(run_ids)
```

You can filtering runs by passing keyword arguments.

```python
run_ids = client.search_run_ids('torch', lr=1e-4, batch_size=32)
print_run_info(run_ids)
```

`Client.search_nested_run_ids()` makes an iterator that returns Run IDs of runs that have a parent run. Optionally, you can filter runs.

```python
run_ids = client.search_nested_run_ids('torch')
print_run_info(run_ids)
```

Note that the `run#0` isn't returned because it was created by `Client.create_run()` directly.


`Client.search_parent_run_ids()` makes an iterator that returns Run IDs of runs that have nested runs. In this case, parent runs are three tasks we made above.

```python
run_ids = client.search_parent_run_ids('torch')
print_run_info(run_ids)
```

### Get functions

`Client.get_run_id()` returns a Run ID of runs you select by run name.

```python
run_id = client.get_run_id('torch', run=0)
print_run_info([run_id])
```

`Client.get_run_ids()` makes an iterator that returns Run IDs of runs you select by run names.

```python
run_ids = client.get_run_ids('torch', task=range(1, 3))
print_run_info(run_ids)
```

`Client.get_nested_run_ids()` makes an iterator that returns Run IDs of runs that have a parent you select by run names.

```python
run_ids = client.get_nested_run_ids('torch', task=range(2))
print_run_info(run_ids)
```

`Client.get_parent_run_id()` returns a Run ID of a run that is refered by a nested run.

```python
run_id = client.get_parent_run_id('torch', run=5)
print_run_info([run_id])
```

### Set function

Sometimes, you may want to change a parent for nested runs. Use `Client.set_parent_run_id()`.

```python
run_ids = client.get_nested_run_ids('torch', task=2)
print_run_info(run_ids)
```

```python
client.set_parent_run_id('torch', run=(0, 2, 3), task=2)
run_ids = client.get_nested_run_ids('torch', task=2)
print_run_info(run_ids)
```

## Next Step

Once you got Run ID(s), you can load a run, a member of a run, or collect results of multiple runs for an ensemble. See [the quickstart](../../quickstart#load-runs-and-results).
