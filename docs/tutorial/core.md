# Ivory Core Entities

{{ ## cache:clear }}

## Client

Ivory has the `Client` class that manages the workflow of machine learning. In this tutorial, we are working with data and model to predict rectangle area. The source module exists under the `examples` directory.

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
  shutil.rmtree('examples/mlruns')
```

First, create a `Client` instance.

```python
import ivory

client = ivory.create_client("examples")  # Set the working directory
client
```

```python
list(client)
```

The first instance is a `Tracker` instance that connects Ivory to [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html). The second instance is named `tuner`. A `Tuner` instance connects Ivory to [Optuna](https://preferred.jp/en/projects/optuna/).

Show files in the working directory `examples`.

```python
import os

os.listdir('examples')
```

`rectangle` is a Python package that contains our examples. YAML files with extension of `.yml` or possibly `.yaml` are parameter files to define a machine learning workflow. Basically, one YAML file is corresponding to one `Experiment` as discussed later, except the `client.yml` file. A YAML file name without the extension becomes an experiment name. `mlruns` is a directory automatically created by MLFlow Tracking in which our trained model and callbacks instances are saved.

The `client.yml` is a configuration file for a `Client` instance. In our case, the file just contains the minimal settings.

#File client.yml {%=/examples/client.yml%}

!!! note
    If you don't need any customization, the YAML file for client is not required. If there is no file for client, Ivory creates a default client with a tracker and tuner. (So, the above file is unnecessary.)

    If you don't need a tracker and/or tuner, for example in debugging, use `ivory.create_client(tracker=False, tuner=False)`.

## Experiment

`Client.create_experiment()` creates an `Experiment` instance. If the `Client` instance has a `tracker`, an experiment of MLFlow Tracking is also created at the same time if it hasn't existed yet. By clicking an icon (<i class="far fa-eye-slash" style="font-size:0.8rem; color: #ff8888;"></i>) in the below cell, you can see the log.

```python
experiment = client.create_experiment('torch')  # Read torch.yml as params.
experiment
```

The ID for this experiment was given by MLFlow Tracking. The `Client.create_experiment()` loads a YAML file corresponding to the first argument from the working directory.

#File torch.yml {%=/examples/torch.yml%}

After loading, the `Experiment` instance setups the parameters for creating runs later. The parameters are stored in the `params` attribute.

```python
experiment.params
```

This is similar to the YAML file we read before, but has been slightly changed.

* Run and experiment keys are inserted.
* Run name is assigned by Ivory Client.
* Experiment ID and Run ID are assigned by MLFlow Tracking.
* Default classes are specified, for example the `ivory.torch.trainer.Trainer` class for a `trainer` instance.

## Run

After setting up an `Experiment` instance, you can create runs with various parameters. Ivory provides several way to configure them as below.

### Default parameters

Calling without arguments creates a run with default parameters.

```python
run = experiment.create_run()
run
```

Here, the ID for this run is assigned by MLFlow Tracking. On the other hand, the name is assigned by Ivory as the form of "`(run class name in lower case)#(run number)`".

### Simple literal (int, float, str)

Passing key-value pairs, you can change the parameters.

```python
run = experiment.create_run(fold=1)
run.datasets.fold
```

But the type of parameter must be equal, otherwise a `ValueError` is raised.

```python
run = experiment.create_run(fold=0.5)
run.datasets.fold
```

### List

A list parameter can be overwritten by passing a new list. Off course you can change the length of the list. The original `hidden_sizes` was `[10, 20]`. Modify it.

```python
run = experiment.create_run(hidden_sizes=[2, 3, 4])
run.model
```

As an alternative way, you can use *0-indexed colon-notation* like below. In this case, pass a dictionary to the first argument, because a colon (`:`) can't be in keyword arguments.

```python
params = {
    "hidden_sizes:0": 10,  # Order is important.
    "hidden_sizes:1": 20,  # Start from 0.
    "hidden_sizes:2": 30,  # No skip. No reverse.
}
run = experiment.create_run(params)
run.model
```

Do you feel this function is unnecessary? This function is prepared for [hyperparameter tuning](../tuning).


In some case, you may want to change elements of list. Use *0-indexed dot-notation*.

```python
params = {"hidden_sizes.1": 5}
run = experiment.create_run(params)
run.model
```

### Duplicated parameter name

Duplicated parameters with the same name are updated together.

```python
run = experiment.create_run(patience=5)
run.scheduler.patience, run.early_stopping.patience
```

This behavior is natural to update the parameters with the same meaning. But in the above example, the patience of early stopping becomes equal to that of scheduler, so the scheduler doesn't work at all.

### Scoping by dots

To specify an individual parameter even if there are other parameters with the same name, use scoping by dots, or *parameter fullname*.

```python
params = {'scheduler.patience': 8, 'early_stopping.patience': 20}
run = experiment.create_run(params)
run.scheduler.patience, run.early_stopping.patience
```

### Object type

Parameters are not limited to a literal such as `int`, `float`, or `str`. For example,

```python
run = experiment.create_run()
run.optimizer
```

```python
run = experiment.create_run({'optimizer.class': 'torch.optim.Adam'})
run.optimizer
```

This means that you can compare optimizer algorithms easily through multiple runs with minimal effort.

### Creating a run from a client

In the above examples, we created runs using the `experiment.create_run()`. In addition, you can do the same thing by `client.create_run()` with an experiment name as the first argument. The following code blocks are equivalent.

#Code
~~~python
experiment = client.create_experiment('torch')
run = experiment.create_run(fold=3)
~~~

#Code
~~~python
run = client.create_run('torch', fold=3)
~~~
