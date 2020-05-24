# Quickstart

Ivory is a lightweight framework for machine learning. It integrates model design, tracking, and hyperparmeter tuning. Ivory uses [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) for tracking and [Optuna](https://preferred.jp/en/projects/optuna/) for hyperparmeter tuning.

The relationship of these libraries is shown in Table {#relationship#}.

#Tab {#relationship#}
Ivory      | MLFlow     | Optuna
-----------|------------|--------
Experiment | Experiment | Study
Run        | Run        | Trial


Using Ivory, you can obtain the both tracking and tuning workflow at one place.

Another key feature of Ivory is its model design. You can write down all of your model structure and tracking/tuning process in one YAML file. It allows us to understand the whole process at a glance.


## Installation

You can install Ivory by a `pip` command.

~~~bash terminal
$ pip install ivory
~~~

## Using a Ivory Client

Ivory has the `Client` class that manages the workflow of any machine learning. Let's create your first `Client` instance. In this quickstart, we are working with examples under the `examples` directory.

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
  shutil.rmtree('examples/mlruns')
  print('deleted')
```

```python
import numpy as np

import ivory

client = ivory.create_client("examples")
client
```

The representation of the `client` shows that it has two objects. Objects that a client has can be accessed by **index notation** or **dot notation**.

```python
client[0]  # or client['tracker'], or client.tracker
```

The first object is a `Tracker` instance which connects Ivory to [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html).

Because a `Client` insctance is an iterable, you can get all of the objects by applying `list` to it.

```python
list(client)
```

The second objects is named `tuner`.

```python
client.tuner
```

A `Tuner` instance connects Ivory to [Optuna: A hyperparameter optimization framework](https://preferred.jp/en/projects/optuna/).

We can customize these objects with a YAML file named `client.yml` under the woking directory.  In our case, the file just contains the minimum settings.

#File client.yml {%=examples/client.yml%}

!!! note
    A YAML file for client is not required. If there is no file for client, Ivory creates a default client with a tracker and without a tuner.

    If you don't need a tracker, use `ivory.create_client(tracker=False)`.

## Create NumPy data

In this quickstart, we try to predict rectangles area from thier width and height using [PyTorch](https://pytorch.org/). First, prepare the data as [NumPy](https://numpy.org/) arrays. In `example.py` under the working directory, a `create_data()` function is defined. The `ivory.create_client()` function automatically inserts the working directory, so that we can import the module.

```python
import example
```

Let's check the `create_data()` function definition and an example output:

#Code example.create_data {{ example.create_data # inspect }}

```python
xy, z = example.create_data(4)
xy
```

```python
z
```

## Set of Data classes

Ivory defines a set of Data classes (`Data`, `Dataset`, `Datasets`, `DataLoaders`). But now, we just introduce the `Data` class.

#Code example.Data {{ example.Data # inspect }}

Here, `kfold_split` function creates a fold-array. In Ivory, fold number = `-1` means their samples are test set.

```python
from ivory.utils.fold import kfold_split  # isort:skip

kfold_split(np.arange(10), n_splits=3)
```

Now, we can get a `Data` instance.

```python
data = example.Data()
data
```

```python
data.get(0)
```

## Define a model

We use a simple MLP model here.

#Code example.Model {{ example.Model # inspect }}

## Parameter file for Run

Ivory defines a run by a YAML file. Here is a full example.

#File torch.yaml {%=examples/torch.yml%}

Let's create a run by `client.create_run()`

```python
run = client.create_run('torch')
run
```

!!! note
    `client.create_run(<name>)` creates an experiment named `<name>` if it hasn't exist yet. By cliking an icon (<i class="far fa-eye-slash" style="font-size:0.8rem; color: #ff8888;"></i>) in the above cell, you can know that.

    Or you can directly create an experiment then make the experiment create a run:

    ~~~python
    experiment = client.create_experiment('torch')
    run = experiment.create_run()
    ~~~

A `Run` instance have a `params` attribute that holds the parameters for the run.

```python
import yaml

print(yaml.dump(run.params, sort_keys=False))
```

This is similar to the YAML file we read before, but slightly changed by the Ivory Client.

* Run and experiment sections are inserted.
* ExperimentID and RunID are assigned by the MLFlow Tracking.
* Default classes are specified, for example `ivory.torch.trainer.Trainer` for a trainer instance.

The `create_run()` method takes keyword arguments to modify these parameters:

```python
run = client.create_run(
  'torch', batch_size=20, hidden_sizes=[40, 50, 60],
)

print('[dataloaders]')
print(yaml.dump(run.params['run']['dataloaders'], sort_keys=False))
print('[model]')
print(yaml.dump(run.params['run']['model'], sort_keys=False))
```

## Train the model

Once you got a run instance, then all you need is to start it.

```python
run = client.create_run('torch')  # Back to the default settings.
run.start()
```

The history of metrics is saved as the `history` attribute of `run.metrics`.

```python
run.metrics.history
```

Also the model output and target are automatically collected.

```python
run.results
```

```python
run.results.val.output[:5]
```

```python
run.results.val.target[:5]
```

## Test the model

Testing a model is as simple as training. Just call `run.start()` with a `test` argument in stead of (default) `train`.

```python
run.start('test')
run.results
```

You can see, `test` results were added.

```python
run.results.test.output[:5]
```

Off course the target values are `np.nan`.

```python
run.results.test.target[:5]
```

## Task for multiple runs

Ivory implements special run type called **Task** which controls multiple nested runs. A task is useful for parameter search or cross validation.

```python
task = client.create_task('torch')
task
```

The `Task` class has two methods to generate multiple runs: `prodcut()` and `chain()`. These two methods have the same functionality as [`itertools`](https://docs.python.org/3/library/itertools.html) of Python starndard library. Let's try cross validation.

```python
runs = task.product(fold=range(4), verbose=0, epochs=3)
runs
```

Like `itertools`'s functions, `Task.prodcut()` and `Task.chain()` return a generator, which yields runs that are configured by different parameters you specified. In this case, this generator will yield 4 runs with a fold number ranging from 0 to 4 for each. A `task` instance doesn't start any training by itself.

!!! note
    You can pass fixed parameters to update the original parameters in the YAML file.

Then start 4 runs by a `for` loop including `run.train()`. Here a `both` argument means execution of test after training.

```python
for run in runs:
    run.start('both')
```

## Collect runs

Our client has a `Tracker` instance. It stores the state of runs in background using the powerful [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html). The `Client` class provides several methods to access the stored runs. For example, `search_run_ids()` returns a generator which yields RunID created by the MLFlow Tracking.

```python
def print_run_info(run_ids):
    for run_id in run_ids:
        print(run_id[:5], client.get_run_name(run_id))
```

```python
run_ids = client.search_run_ids('torch')
print_run_info(run_ids)
```

For filtering, add key-value pairs.

```python
run_ids = client.search_run_ids('torch', fold=0, exclude_parent=True)
print_run_info(run_ids)
```

```python
run_ids = client.search_run_ids('torch', parent_run_id=task.id)
print_run_info(run_ids)
```

!!! note
    * `exclude_parent`: If True, parent runs are excluded.
    * `parent_run_id`: If specified, nested runs are returned only if it has the parent.

`client.get_run_id()` and `client.get_run_ids()` fetch RunID from run name, more strictly, (run class name in lowercase) plus (run number).

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

The Ivory `Client` class can load runs. First select RunID(s) to load. We want to perform cross validation here, so that we need a run collection created by a task. In this case, we can use `client.get_nested_run_ids()`. Why don't use `client.search_run_ids` as we did above? Because generally we can't write down a very long RunID. On the ohter hand, a run name is easy to manege and write.

```python
run_ids = list(client.get_nested_run_ids('torch', task=0))
print_run_info(run_ids)
```

Let's load the latest run.

```python
run = client.load_run(run_ids[0])
run
```

!!! note
    `client.load_run()` doesn't require an experiment name, because RunID is unique among MLFlow Tracking.

As you expected, the fold number is 3.

```python
run.dataloaders.fold
```

We obtained the trained model.

```python
run.model.eval()
```

```python
import torch

index, input, target = next(iter(run.dataloaders.val))
with torch.no_grad():
    output = run.model(input)
print('[output]')
print(output)
print('[target]')
print(target)
```

If you don't need a whole run instance, `client.load_instance()` is better choice to save time and memory.

```python
results = client.load_instance(run_ids[0], 'results')
results
```

```python
for mode in ['train', 'val', 'test']:
    print(mode, results[mode].output.shape)
```

For cross validation, we need 4 runs. (`n_splits` is 5 but we used the last fold for dummy test set.) To load multiple run's results. Ivory Client provides
a convenient method.

```python
results = client.load_results(run_ids, verbose=False)
results
```

```python
for mode in ['val', 'test']:
    print(mode, results[mode].output.shape)
```

!!! note
    `client.load_results` drops train set for saving memory.

The lengths of validation set and test set are both 800 (200 times 4). But be careful about the test set. The length of unique samples is 200 (one fold size).

```python
import numpy as np

len(np.unique(results.val.index)), len(np.unique(results.test.index))
```

Usually, duplicated samples are averaged. `Results.mean()` method performs this mean reduction and returns a newly created `Rusults` instance.

```python
reduced_results = results.mean()
for mode in ['val', 'test']:
    print(mode, reduced_results[mode].output.shape)
```

Check the consistency.

```python
index = results.test.index
index_0 = index[0]
x = results.test.output[index == index_0]
print('[results]')
print(x)
print(np.mean(x))

index = reduced_results.test.index
x = reduced_results.test.output[index == index_0]
print('[reduced_results]')
print(x)
```
