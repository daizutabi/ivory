# Tracking UI

Ivory uses [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html) for the workflow tracking and model saving. For this feature, the `Client` instace has to have a `Tracker` instance.

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
run = client.create_run('torch')
run.start('both')
```

```python
task = client.create_task('torch')
runs = task.product(fold=range(3), verbose=0)
for run in runs:
    run.start('both')
```

```python
task = client.create_task('torch')
runs = task.chain(lr=[1e-4, 1e-3], batch_size=[16, 32], verbose=0)
for run in runs:
    run.start('both')
```

```python
from ivory.utils.range import Range

study = client.create_study('torch', lr=Range(1e-5, 1e-3, log=True))
study.optimize(n_trials=5, verbose=0)
```

## Tracking UI

Optionally, you can update missing parameters:

```python
client.update_params('torch')
```

In a terminal, move to the working directory (`examples`), then run

```bash
$ ivory ui
```

You can view the UI using URL http://localhost:5000 in your browser.

#Tab A collection of runs. Parameters, metrics, tags are logged.
![png](/img/img1.png)

You can compare the training results among runs.

#Fig Comparison of training curves
![png](/img/img2.png)

See also [the official MLFlow documentation](https://mlflow.org/docs/latest/tracking.html#tracking-ui).
