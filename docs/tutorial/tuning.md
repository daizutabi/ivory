# Hyperparameter Tuning

{{ # cache:clear }}

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
    shutil.rmtree('examples/mlruns')
```

## Suggest Function

To optimize a set of hyperparameters, define a *suggest function*. Here are  example functions.

#File rectangle/suggest.py {%=/examples/rectangle/suggest.py%}

A suggest function must take a [`trial`](https://optuna.readthedocs.io/en/latest/reference/trial.html#) as the first argument but you can add arbitrary arguments if you need. See also [the Optuna offical documentation]( https://optuna.readthedocs.io/en/latest/tutorial/configurations.html) for more details.

!!! note
    In the `suggest_hidden_sizes()` function, we use [*0-indexed colon-notation*](../core#list), because Optuna doesn't suggest a list itself but its element.

These suggest functions don't return any parameters. The only work of suggest functions is to make the `Trial` instance suggest parameters. Suggested parameters are stored in the `Trial` instance, so that nothing is needed from suggest functions.

Note that the objective function in Optuna has only one `trial` argument, so that we have to use the `functools.partial()` function that returns a *pure* suggest function.

```python
from functools import partial
from rectangle.suggest import suggest_lr, suggest_hidden_sizes

lr = partial(suggest_lr, min=1e-5, max=1e-2)
hidden_sizes = partial(suggest_hidden_sizes, max_num_layers=3)
```

## Study

Ivory implements a special run type called **Study** which controls hyperparameter tuning using [Optuna](https://preferred.jp/en/projects/optuna/). The `Study` class is a subclass of the [`Task`](../task) class so that the same [tracking system](../task#tracking) can be used.


```python
import ivory

client = ivory.create_client("examples")  # Set the working directory
study_lr = client.create_study('torch', lr=lr)
study_hs = client.create_study('torch', hidden_sizes=hidden_sizes)
study_lr
```

## Objective

The `ivory.core.objective.Objective` class provides *objective functions* that return a score to minimize or maximize. But you don't need to know about the `Objective` class in details. Ivory builds an objective function from a suggest function and provides it to Optuna so that Optuna can optimize the parameters.

A `Study` instance has an `Objective` instance.

```python
study_lr.objective
```

```python
study_hs.objective
```

## Optimization

Then "optimize" the learning rate and hidden sizes just for fun.

```python
optuna_study_lr = study_lr.optimize(n_trials=3, fold=3, epochs=3)
```

```python
optuna_study_hs = study_hs.optimize(n_trials=3, epochs=3)
```

!!! note
    By cliking an icon (<i class="far fa-eye-slash" style="font-size:0.8rem; color: #ff8888;"></i>) in the above cells, you can see the Optuna's log.

The returned value of the `study.optimize()` is an Optuna's `Study` instance (not Ivory's one).

```python
optuna_study_lr
```

The `Study` instance is named after the experiment name, suggest name, and run name.

```python
optuna_study_lr.study_name
```

In [user attributes](https://optuna.readthedocs.io/en/latest/tutorial/attributes.html) that Optuna's `Study` and `Trial` instances provide, RunID is saved.

```python
optuna_study_lr.user_attrs
```

```python
optuna_study_lr.trials[0].user_attrs
```

On the other hand, MLFlow Tracking's run (not Ivory's one) has a tag to refer Optuna's study and trial.

```python
mlflow_client = client.tracker.client
mlflow_client
```

```python
run_id = optuna_study_lr.user_attrs['run_id']
run = mlflow_client.get_run(run_id)  
run.data.tags['study_name']
```

```python
run_id = optuna_study_lr.trials[0].user_attrs['run_id']
run = mlflow_client.get_run(run_id)  
run.data.tags['trial_number']
```

You may have a question. How does Optuna optimize the parameters without any score? The answer is the `Monitor` instance. An `Objective` instance gets the monitoring score from `run.monitor` and sends it to Optuna so that Optuna can determine the next suggestion. All you need is to make your `Run` instance have a `Monitor` instance. Check the YAML parameter file:

#File torch.yml {%=/examples/torch.yml%}

The `Monitor` instance monitors `val_loss` and the default mode is `min` (smaller is better). If your monitor  is accuracy, for example, set the monitor like this:

~~~yaml
monitor:
  metric: accuracy
  mode: max
~~~

## Parametric Optimization

Again read the suggest functions.

#File rectangle/suggest.py {%=/examples/rectangle/suggest.py%}

The `suggest_hidden_sizes()` function has some logic but the `suggest_lr()` function is too simple to define a function. You may not want to write such a function. Ivory can do that for you. You can pass iterable(s) to the `client.create_study()` function instead of a callable

```python
study = client.create_study('torch', lr=(1e-3, 1e-2))  # `tuple` for range
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

```python
from ivory.utils.range import Range  # `Range` for log scale.

study = client.create_study('torch', lr=Range(1e-3, 1e-2, log=True))
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

```python
params = {'hidden_sizes.0': range(10, 20)}  # `range` for integer range.
study = client.create_study('torch', params)
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

```python
params = {'hidden_sizes.0': [10, 20, 30]}  # `list` for choice.
study = client.create_study('torch', params)
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

```python
# Product
params = {('hidden_sizes.1', 'lr'): (range(10, 20), Range(1e-4, 1e-3))}
study = client.create_study('torch', params)
_ = study.optimize(n_trials=10, epochs=1, verbose=0)
```

!!! note
    You may feel that "`params = {'hidden_sizes.1': range(10, 20), 'lr': Range(1e-4, 1e-3)}`" must be better, but the above style is intentional.
