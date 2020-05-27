# Ivory Core Entities

## Client

Ivory has the `Client` class that manages the workflow of machine learning. In this tutorial, we are working with data and model to predict rectangle area. The source module exists under the `examples` directory.

{{ # cache:clear }}

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

The first object is a `Tracker` instance which connects Ivory to [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html).

The second objects is named `tuner`. A `Tuner` instance connects Ivory to [Optuna: A hyperparameter optimization framework](https://preferred.jp/en/projects/optuna/).

Show the files in the working directory `examples`.

```python
import os

os.listdir('examples')
```

`rectangle` is a Python package that contains our examples. YAML files with extension of `.yml` or possibly `.yaml` are parameter files to define a machine learning workflow. Basically, one YAML file is corresponding to one `Experiment` as discussed later, except the `client.yml` file. A YAML file name without the extension becomes an experiment name. `mlruns` is a directory automatically created by the MLFlow Tracking in which our trained model and callbacks instances are saved.

The `client.yml` is a configuration file for a `Client` instance. In our case, the file just contains the minimum settings.

#File client.yml {%=examples/client.yml%}

!!! note
    A YAML file for client is not required. If there is no file for client, Ivory creates a default client with a tracker and without a tuner.

    If you don't need a tracker, for example in debugging, use `ivory.create_client(tracker=False)`.

## Experiment

The `Client.create_experiment()` function creates an `Experiment` instance. If the `Client` instance has a `tracker`, an experiment of the MLFlow Tracking is also created at the same time if it hasn't existed yet. By cliking an icon (<i class="far fa-eye-slash" style="font-size:0.8rem; color: #ff8888;"></i>) in the below cell, you can see the log.

```python
experiment = client.create_experiment('torch')  # Read torch.yml as params.
experiment
```

The ID for this experiment was given by the MLFlow Tracking. The `Client.create_experiment()` function loads a corresponding YAML file to the first argument from the working directory.

#File torch.yml {%=examples/torch.yml%}

After loading, the `Experiment` instance setups the parameters for creating runs later. The parameters are stored in the `params` attribute.

```python
experiment.params
```

This is similar to the YAML file, but is slightly changed by the Ivory Client.

* Run and experiment sections are inserted.
* ExperimentID and RunID are assigned by the MLFlow Tracking.
* Default classes are specified, for example `ivory.torch.trainer.Trainer` for a trainer instance.

## Run

After setting up an `Experiment` instance, you can create runs with various parameter.

**Default parameters**

Calling without arguments creates a run with default parameters.

```python
run = experiment.create_run()
run
```

Here, the ID for this run was given by the MLFlow Tracking. On the other hand, the name is given by Ivory as a form of "`(run class name in lower case)#(run number)`".


**Parameter configuration**

Passing key-value pairs, you can change the parameters.

```python
run = experiment.create_run(fold=1)
run.dataloaders.fold
```

But the type of parameter must be equal, otherwise a `ValueError` is raised.

```python
run = experiment.create_run(fold=0.5)
run.dataloaders.fold
```

**Duplicated parameter**

Duplicated parameters are updated together.

```python
run = experiment.create_run(patience=5)
run.scheduler.patience, run.early_stopping.patience
```

**Scoping by dots**

To specify an individual parameter, use scoping by dots. In this case, pass a dictionary with string type keys with dots to the first argument.

```python
update = {'scheduler.patience': 8, 'early_stopping.patience': 20}
run = experiment.create_run(update)
run.scheduler.patience, run.early_stopping.patience
```

**Object type**

Parameters are not limited to a literal such as `int`, `float`, or `str`. For example,

```python
run = experiment.create_run()
run.optimizer
```

```python
run = experiment.create_run({'optimizer.class': 'torch.optim.Adam'})
run.optimizer
```

This means that you can compare optimizer algorithms easily through multiple runs with minimul effort.

In the previous examples, we created runs using the `experiment.create_run()` method. In addtion, you can do the same thing by `client.create_run()` with an experiment name as the first argument. The following code blocks are qeuivalent.

#Code
~~~python
experiment = client.create_experiment('torch')
run = experiment.create_run(fold=3)
~~~

#Code
~~~python
run = client.create_run('torch', fold=3)
~~~

## Task for multiple runs

Ivory implements a special run type called **Task** which controls multiple nested runs. A task is useful for parameter search or cross validation.

```python
# task = experiment.create_task()  #  Alternative method.
task = client.create_task('torch')
task
```

The `Task` class has two methods to generate multiple runs: `prodcut()` and `chain()`. These two methods have the same functionality as [`itertools`](https://docs.python.org/3/library/itertools.html) of Python starndard library.

```python
runs = task.product(fold=range(2), lr=[1e-4, 1e-3])
runs
```
