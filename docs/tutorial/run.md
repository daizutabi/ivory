# Run

Now we reached the time to invoke a `Run`. The latest `params.yaml` is like below:

#File params_3.yaml {%=params_3.yaml%}

Umm... `run` isn't defined anywhere. Because a `run` instance is created by an `experiment` instance dynamically, we don't need define a `run` instance in a parameters file. Instead of it, `experiment` has a field of `run_class` in order to determine which `Run` class sould be created.

First, create an experiment instance.

```python
import ivory

experiment = ivory.create_experiment('params_3.yaml')
experiment
```

You created an `Experiment` instance. It says that its name is "ready" and its `run_class` is `ivory.torch.Run`, now. In addtion, the `shared` field becomes `['input', 'target']`. The experiment can detect that `target` should be also shared as well as `input` if `input` is shared.

`create_data` and `Model` are in global space so we need define here again.

```python
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import ivory
from ivory.utils import kfold_split

def create_data(num_samples=1000):
    """Returns a tuple of (input, target). Target has fold information."""
    x = 4 * np.random.rand(num_samples, 2) + 1
    x = x.astype(np.float32)
    noises = 0.1 * (np.random.rand(2, num_samples) - 0.5)
    df = pd.DataFrame(x, columns=["width", "height"])
    df["area"] = (df.width + noises[0]) * (df.height + noises[1])
    df.area = df.area.astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df[["width", "height"]], df[["fold", "area"]]


class Model(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        it = zip([2] + hidden_sizes, hidden_sizes + [1])
        for in_features, out_features in it:
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
```

Next, you can start the `Experiment`.

```python
experiment.start()
experiment
```

Shared ojbects are stored at `default` field. (`experiment` itself is also shared.)

```python
experiment.default.keys()
```

Create a `Run`.

```python
run = experiment.create_run()
run
```

Run has `params` field that defines the hyper parameters space.

```python
run.params
```

All the first-level key in this `params` dictionary are set as `run`s attributes. Because `Run` is iterable, you can scan theses attributes:

```python
list(run)
```

Take a look at `model`.

```python
run.model
```

If you give a new dictionary to `create_run` method, you can get a different run with different hyper parameters.

```python
run2 = experiment.create_run({"model.hidden_sizes": [3, 4, 5]})
run2.model
```

Yes. The layer structure are changed. `run2` is a new `Run` instance which is independent of the first `run`. But, `input` and `target` are shared.

```python
run.input is run2.input, run.target is run2.target
```

This time `run` can start successfully.

```python
run.start()
```

After a run,  you can check the metrics.

```python
run.metrics.best_epoch, run.metrics.best_score
```

```python
history = run.metrics.history
history
```

```python
import matplotlib.pyplot as plt

plt.plot(history.index, history.loss, marker="o", label="loss")
plt.plot(history.index, history.val_loss, marker="s", label="val_loss")
plt.legend()
plt.yscale("log")
```

Best validation output of model is also stored.

```python
run.metrics.best_output.head()
```

Here. the index is  corresponding to that of the target.

```python
print(len(run.target))
run.target.head()
```

```python
df = run.target.join(run.metrics.best_output, how='inner')
df.fold.unique()
```

```python
plt.scatter(df.area, df.output)
```
