# Hyperparameter Tuning

{{ ## cache:clear }}

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
    shutil.rmtree('examples/mlruns')
```

## Suggest Function

To optimize a set of hyperparameters, define a *suggest function*. Here are  example functions.

#File rectangle/suggest.py {%=/examples/rectangle/suggest.py%}

A suggest function must take a `trial` (an instance of [`Trial`](https://optuna.readthedocs.io/en/latest/reference/trial.html#)) as the first argument but you can add arbitrary arguments if you need. For more details about what the `Trial` can do, see [the Optuna offical documentation]( https://optuna.readthedocs.io/en/latest/tutorial/configurations.html).

!!! note
    In the `suggest_hidden_sizes()` function, we use [*0-indexed colon-notation*](../core#list), because Optuna doesn't suggest a list itself but its element.

These suggest functions don't return any parameters. The only work of suggest functions is to make the `Trial` instance suggest parameters. Suggested parameters are stored in the `Trial` instance, so that nothing is needed from suggest functions.

Note that [an objective function in Optuna](https://optuna.readthedocs.io/en/latest/tutorial/first.html) has only one `trial` argument, so that we have to use the `functools.partial()` function to make a *pure* suggest function.

```python
from functools import partial
from rectangle.suggest import suggest_lr, suggest_hidden_sizes

lr = partial(suggest_lr, min=1e-5, max=1e-2)
hidden_sizes = partial(suggest_hidden_sizes, max_num_layers=3)
```

## Study

Ivory implements a special run type called **Study** which controls hyperparameter tuning using [Optuna](https://preferred.jp/en/projects/optuna/).

```python
import ivory

client = ivory.create_client("examples")  # Set the working directory
study_lr = client.create_study('torch', lr=lr)
study_hs = client.create_study('torch', hidden_sizes=hidden_sizes)
study_lr
```

In the `client.create_study()` function, you can pass keyword arguments in which a key is a suggest names and a value is a pure suggest functions.


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

### tuple, range, Range

A tuple, range, or [Range](../task#range) instance represent parameter ranges.

```python
study = client.create_study('torch', lr=(1e-3, 1e-2))
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

In the above cell, `lr=Range(1e-3, 1e-2)` also works. For integer parameters, you can use normal `range` as well as `tuple` or `Range`.

```python
params = {'hidden_sizes.0': range(10, 20)}
study = client.create_study('torch', params)
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

You can specify a step

```python
params = {'hidden_sizes.0': range(10, 20, 3)}
study = client.create_study('torch', params)
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

If you need sampling in log scale, use `Range` with `log=True`.

```python
from ivory.utils.range import Range

study = client.create_study('torch', lr=Range(1e-3, 1e-2, log=True))
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

### list

A list represents parameter choice.

```python
params = {'hidden_sizes.0': [10, 20, 30]}
study = client.create_study('torch', params)
_ = study.optimize(n_trials=5, epochs=1, verbose=0)
```

### Product

If a key and value are tuples, the entry means cartesian product of suggest functions like [`Task.product()`](../task#product).

```python
params = {('hidden_sizes', 'lr'): (hidden_sizes, Range(1e-4, 1e-3))}
study = client.create_study('torch', params)
optuna_study = study.optimize(n_trials=10, epochs=1, verbose=0)
```

!!! note
    You can mix suggest funtions and parametric optimization.


!!! note
    You may feel that "`params = {'hidden_sizes.1': hidden_sizes, 'lr': Range(1e-4, 1e-3)}`" must be better, but the above style is intentional.

In parametric optimization, the name of Optuna's `Study` instance is *dot-joint style*:

```python
optuna_study.study_name
```

## Study from YAML file

As a normal `Run`, a `Study` instance also can be created from a YAML file. For this, pass an extra keyword argument to the `client.create_experiment()` function. The key is the instance name (in this case `study`) and value is a YAML file name without the extension.

```python
experiment = client.create_experiment('torch', study='study')
experiment
```

Here is the contents of `study.yml` file.

#File study.yml {%=/examples/study.yml[3:]%}

Suggest functions should be callable, `hidden_sizes` uses `def` keyword. On the other hand, `lr` is just one line. If a suggest funtion can be called without additional parameters, you can omit the `def` keyword. Using this experiment, create `Study` instances.

```python
study_lr = client.create_study('torch', 'lr')
study_lr.objective
```

```python
study_hs = client.create_study('torch', 'hidden_sizes')
study_hs.objective
```

```python
study_hs.objective.hidden_sizes
```

For `min_size` and `max_size`, default values are inspected from the signature.

```python
study_lr.optimize(n_trials=3, epochs=3, verbose=0)
```

## Pruning

Optuna provides [the pruning functionality](https://optuna.readthedocs.io/en/latest/tutorial/pruning.html). Ivory can uses this feature seamlessly.

Here is the updated contents of `study.yml` file.

#File study.yml {%=/examples/study.yml%}

The `Tuner` instance has Optuna's `MedianPruner`. (Off course, you can use [other pruners](https://optuna.readthedocs.io/en/latest/reference/pruners.html).) A `Study` instance give an `ivory.callbacks.Pruning` instance to a run when the run is created, then with Ivory's [callback system](../callbacks), the `Pruning` instance communicates with Optuna.


!!! note
    Pruning is supported for PyTorch and TensorFlow now.
