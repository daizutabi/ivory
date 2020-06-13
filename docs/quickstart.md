# Quickstart

{{ ## cache:clear }}

## Installation

Install Ivory using `pip`.

~~~bash terminal
$ pip install ivory
~~~

## Ivory Client

Ivory has the `Client` class that manages the workflow of machine learning. Let's create your first `Client` instance. In this quickstart, we are working with examples under the `examples` directory. Pass `examples` to the first argument of `ivory.create_client()`:

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
  shutil.rmtree('examples/mlruns')
```

```python
import ivory

client = ivory.create_client("examples")
client
```

The representation of the `client` shows that it has two instances. These instances can be accessed by *index notation* or *dot notation*.

```python
client[0]  # or client['tracker'], or client.tracker
```

The first instance is a `Tracker` instance that connects Ivory to [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html).

Because a `Client` instance is an iterable, you can get all of the instances by applying `list()` to it.

```python
list(client)
```

The second instance is named `tuner`.

```python
client.tuner
```

A `Tuner` instance connects Ivory to [Optuna: A hyperparameter optimization framework](https://preferred.jp/en/projects/optuna/).

We can customize these instances with a YAML file named `client.yml` under the working directory.  In our case, the file just contains the minimal settings.

#File client.yml {%=/examples/client.yml%}

!!! note
    If you don't need any customization, the YAML file for client is not required. If there is no file for client, Ivory creates a default client with a tracker and tuner. (So, the above file is unnecessary.)

    If you don't need a tracker and/or tuner, for example in debugging, use `ivory.create_client(tracker=False, tuner=False)`.

## Create NumPy data

In this quickstart, we try to predict rectangles area from their width and height using [PyTorch](https://pytorch.org/). First, prepare the data as [NumPy](https://numpy.org/) arrays. In `rectangle/data.py` under the working directory, `create_data()` is defined. The `ivory.create_client()` automatically inserts the working directory to `sys.path`, so that we can import the module regardless of the current directory.

Let's check the `create_data()` code and an example output:

```python hide
import rectangle.data
```

#File rectangle/data.py {%=/examples/rectangle/data.py%}

```python
import rectangle.data

xy, z = rectangle.data.create_data(4)
xy
```

```python
z
```

`ivory.utils.fold.kfold_split()` creates a fold array.

```python
import numpy as np
from ivory.utils.fold import kfold_split

kfold_split(np.arange(10), n_splits=3)
```

## Set of Data Classes

Ivory defines a set of base classes for data (`Data`, `Dataset`, `Datasets`, and `DataLoaders`) that user's custom classes can inherit. But now, we use the `Data` only.

Now, we can get a `rectangle.data.Data` instance.

```python
data = rectangle.data.Data()
data
```

```python
data.get(0)  # get data of index = 0.
```

The returned value is a tuple of (index, input, target). Ivory always keeps data index so that we can know where a sample comes from.

## Define a model

We use a simple MLP model. Note that the number of hidden layers and the size of each hidden layer are customizable.

```python hide
import rectangle.torch
```

#File rectangle/torch.py {%=/examples/rectangle/torch.py%}

## Parameter file for Run

Ivory configures a run using a YAML file. Here is a full example.

#File torch.yaml {%=/examples/torch.yml%}

Let's create a run calling the `Client.create_run()`.

```python
run = client.create_run('torch')
run
```

!!! note
    `Client.create_run(<name>)` creates an experiment named `<name>` if it hasn't existed yet. By clicking an icon (<i class="far fa-eye-slash" style="font-size:0.8rem; color: #ff8888;"></i>) in the above cell, you can see the log.

    Or you can directly create an experiment then make the experiment create a run:

    ~~~python
    experiment = client.create_experiment('torch')
    run = experiment.create_run()
    ~~~

A `Run` instance have an attribute `params` that holds the parameters for the run.

```python
import yaml

print(yaml.dump(run.params, sort_keys=False))
```

This is similar to the YAML file we read before, but has been slightly changed.

* Run and experiment keys are inserted.
* Run name is assigned by Ivory Client.
* Experiment ID and Run ID are assigned by MLFlow Tracking.
* Default classes are specified, for example the `ivory.torch.trainer.Trainer` class for a `trainer` instance.

The `Client.create_run()` can take keyword arguments to modify these parameters:

```python
run = client.create_run(
  'torch', fold=3, hidden_sizes=[40, 50, 60],
)

print('[datasets]')
print(yaml.dump(run.params['run']['datasets'], sort_keys=False))
print('[model]')
print(yaml.dump(run.params['run']['model'], sort_keys=False))
```

## Train a model

Once you got a run instance, then all you need is to start it.

```python
run = client.create_run('torch')  # Back to the default settings.
run.start()
```

The history of metrics is saved as the `history` attribute of a `run.metrics` instance.

```python
run.metrics.history
```

```python
run.metrics.history.val_loss
```

Also the model output and target are automatically collected in a `run.results` instance.

```python
run.results
```

```python
run.results.val.output[:5]
```

```python
run.results.val.target[:5]
```

## Test a model

Testing a model is as simple as training. Just call `Run.start('test')` instead of a (default) `'train'` argument.

```python
run.start('test')
run.results
```

As you can see, `test` results were added.

```python
run.results.test.output[:5]
```

Off course the target values for the test data are `np.nan`.

```python
run.results.test.target[:5]
```

## Task for multiple runs

Ivory implements a special run type called **Task** that controls multiple nested runs. A task is useful for parameter search or cross validation.

```python
task = client.create_task('torch')
task
```

The `Task` class has two functions to generate multiple runs: `Task.prodcut()` and `Task.chain()`. These two functions have the same functionality as [`itertools`](https://docs.python.org/3/library/itertools.html) of Python starndard library. Let's try to perform cross validation.

```python
runs = task.product(fold=range(4), verbose=0, epochs=3)
runs
```

Like `itertools`'s functions, `Task.prodcut()` and `Task.chain()` return a generator, which yields runs that are configured by different parameters you specify. In this case, this generator will yield 4 runs with a fold number ranging from 0 to 3 for each. A `task` instance doesn't start any training by itself. In addition, you can pass fixed parameters to update the original parameters in the YAML file.

Then start 4 runs by a `for` loop including `run.start('both')`. Here `'both'` means successive test after training.

```python
for run in runs:
    run.start('both')
```

## Collect runs

Our client has a `Tracker` instance. It stores the state of runs in background using MLFlow Tracking. The `Client` provides several functions to access the stored runs. For example, `Client.search_run_ids()` returns a generator that yields Run ID assigned by MLFlow Tracking.

```python
# A helper function.
def print_run_info(run_ids):
    for run_id in run_ids:
        print(run_id[:5], client.get_run_name(run_id))
```

```python
run_ids = client.search_run_ids('torch')  # Yields all runs of `torch`.
print_run_info(run_ids)
```

For filtering, add key-value pairs.

```python
# If `exclude_parent` is True, parent runs are excluded.
run_ids = client.search_run_ids('torch', fold=0, exclude_parent=True)
print_run_info(run_ids)
```

```python
# If `parent_run_id` is specified, nested runs with the parent are returned.
run_ids = client.search_run_ids('torch', parent_run_id=task.id)
print_run_info(run_ids)
```

`Client.get_run_id()` and `Client.get_run_ids()` fetch Run ID from run name, more strictly, a key-value pair of (run class name in lower case, run number).

```python
run_ids = [client.get_run_id('torch', run=0),
           client.get_run_id('torch', task=0)]
print_run_info(run_ids)
```

```python
run_ids = client.get_run_ids('torch', run=range(2, 4))
print_run_info(run_ids)
```

## Load runs and results

A `Client` instance can load runs. First select Run ID(s) to load. We want to perform cross validation here, so that we need a run collection created by the `task#0`. In this case, we can use `Client.get_nested_run_ids()`. Why don't we use `Client.search_run_ids()` as we did above? Because we don't have an easy way to get a very long Run ID after we restart a Python session and lose the `Task` instance. On the other hand, a run name is easy to manage and write.

```python
# Assume that we restarted a session so we have no run instances now.
run_ids = list(client.get_nested_run_ids('torch', task=0))
print_run_info(run_ids)
```

Let's load the latest run.

```python
run = client.load_run(run_ids[0])
run
```

Note that the `Client.load_run()` doesn't require an experiment name because Run ID is [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier).

As you expected, the fold number is 3.

```python
run.datasets.fold
```

By loading a run, we obtain the *pretrained* model.

```python
run.model.eval()
```

```python
import torch

index, input, target = run.datasets.val[:5]
with torch.no_grad():
    output = run.model(torch.tensor(input))
print('[output]')
print(output.numpy())
print('[target]')
print(target)
```

If you don't need a whole run instance, `Client.load_instance()` is a better choice to save time and memory.

```python
results = client.load_instance(run_ids[0], 'results')
results
```

```python
for mode, result in results.items():
    print(mode, result.output.shape)
```

For cross validation, we need 4 runs. In order to load multiple run's results at the same time, the Ivory `Client` provides a convenient function.

```python
results = client.load_results(run_ids, verbose=False)  # No progress bar.
results
```

```python
for mode, result in results.items():
    print(mode, result.output.shape)
```

!!! note
    `Client.load_results()` drops train data for saving memory.

The lengths of the validation and test data are both 800 (200 times 4). But be careful about the test data. The length of unique samples should be 200 (one fold size).

```python
import numpy as np

len(np.unique(results.val.index)), len(np.unique(results.test.index))
```

Usually, duplicated samples in test data are averaged for ensembling. `Results.mean()` performs this *mean reduction* and returns a newly created `Rusults` instance.

```python
reduced_results = results.mean()
for mode, result in reduced_results.items():
    print(mode, result.output.shape)
```

Compare these two results.

```python
index = results.test.index
index_0 = index[0]
x = results.test.output[index == index_0]
print('[results]')
print(x)
print("-> mean:", np.mean(x))

index = reduced_results.test.index
x = reduced_results.test.output[index == index_0]
print('[reduced_results]')
print(x)
```

For convenience, The `Client.load_results()` has a `reduction` keyword argument.

```python
results = client.load_results(run_ids, reduction='mean', verbose=False)
results
```

```python
for mode, result in results.items():
    print(mode, result.output.shape)
```

The cross validation (CV) score can be calculated as follows:

```python
true = results.val.target
pred = results.val.output
np.mean(np.sqrt((true - pred) ** 2))  # Use any function for your metric.
```

And we got prediction for the test data using 4 MLP models.

```python
results.test.output[:5]
```

## Summary

In this quickstart, we learned how to use the Ivory library to perform machine learning workflow. For more details see the Tutorial.
