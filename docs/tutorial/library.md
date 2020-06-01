# Library Comparison

{{ ## cache:clear }}

So far, we have used PyTorch in this tutorial, but Ivory can perform machine learning with other libraries.

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
    shutil.rmtree('examples/mlruns')
```

## Base Parameter File

Before examples, we write two base or *template* parameter files, which are extended by other parameter files later.

#File A base parameter YAML file for various libraries (data.yml) {%=/examples/data.yml%}

#File A base parameter YAML file for various libraries (base.yml) {%=/examples/base.yml%} {#base#}

In `base.yml`, the first line "`extends: data`" means that the file extends (or includes, in this case) `data.yml`.

## Neural Network Libraries

In this section we compare three neural network libraries ([TensorFlow](https://www.tensorflow.org/), [NNabla](https://nnabla.org/), and [PyTorch](https://pytorch.org/)), and show that using different libraries on the same problem is straightforward.

```python
import tensorflow
import nnabla
import torch
print(tensorflow.__version__)
print(nnabla.__version__)
print(torch.__version__)
```

First define models:

#File A Model definition in TensorFlow (rectangle/tensorflow.py) {%=/examples/rectangle/tensorflow.py%}

#File A Model definition in NNabla (rectangle/nnabla.py) {%=/examples/rectangle/nnabla.py%}

#File A Model definition in PyTorch (rectangle/torch.py) {%=/examples/rectangle/torch.py%}

For simplicity, the TensorFlow model is defined by using the `keras.Sequential()`, so that we call the `create_model()` to get the model.

Next, write parameter YAML files:

#File A parameter YAML file for TensorFlow (tensorflow.yml) {%=/examples/tensorflow.yml%}

#File A parameter YAML file for NNabla (nnabla.yml) {%=/examples/nnabla.yml%}

#File A parameter YAML fine for PyTorch (torch2.yml) {%=/examples/torch2.yml%}

These YAML files are very similar. The only difference is that, in PyTorch, an optimizer needs model parameters at the time of instantiation.

!!! note
    The `model` for TensorFlow is a function. A new `call` key is used. (But you can stil use `class`, or `call` for a class, vice versa, because both a class and function are *callable*.)

Next, create three runs.

```python
import ivory

client = ivory.create_client("examples")
run_tf = client.create_run('tensorflow')
run_nn = client.create_run('nnabla')
run_torch = client.create_run('torch2')
```

For comparison, equalize initial parameters.

```python
import torch

# These three lines are only needed for this example.
run, trainer = run_nn, run_nn.trainer
run.model.build(trainer.loss, run.datasets.train, trainer.batch_size)
run.optimizer.set_parameters(run.model.parameters())

ws_tf = run_tf.model.weights
ws_nn = run_nn.model.parameters().values()
ws_torch = run_torch.model.parameters()
for w_tf, w_nn, w_torch in zip(ws_tf, ws_nn, ws_torch):
    w_nn.data.data = w_tf.numpy()
    w_torch.data = torch.tensor(w_tf.numpy().T)
```

Then, start the runs.

```python
run_tf.start('both')  # Slower due to usage of GPU for a simple network.
```

```python
run_nn.start('both')
```

```python
run_torch.start('both')
```

Metrics during training are almost same. Visualize the results:

```python
import matplotlib.pyplot as plt

# A helper function
def plot(run):
    dataset = run.results.val
    plt.scatter(dataset.target.reshape(-1), dataset.output.reshape(-1))
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.xlabel('Target values')
    plt.ylabel('Predicted values')

for run in [run_tf, run_nn, run_torch]:
    plot(run)
```

Actual outputs are like below:

```python
x = run_tf.datasets.val[:5][1]
run_tf.model.predict(x)
```

```python
x = run_nn.datasets.val[:5][1]
run_nn.model(x)
```

```python
x = run_torch.datasets.val[:5][1]
run_torch.model(torch.tensor(x))
```

You can *ensemble* these results, although this is meaningless in this example.

```python
from ivory.callbacks.results import concatenate

results = concatenate(run.results for run in [run_tf, run_nn, run_torch])
index = results.val.index.argsort()
results.val.output[index[:15]]
```

```python
reduced_results = results.mean()
reduced_results.val.output[:5]
```

## Scikit-learn

Ivory can optimize various [scikit-learn](https://scikit-learn.org/stable/index.html)'s estimators. Before showing some examples, we need reshape the target array.

#File A base parameter YAML file for various estimators (data2.yml) {%=/examples/data2.yml%}

The `dataset` has a `transform` argument. This function reshapes the target array to match the shape for scikit-learn estimators (1D from 2D).

{{ import rectangle.data }}

#Code rectangle.data.transform() {{ rectangle.data.transform # inspect }}

### RandomForestRegressor

#File A parameter YAML file for RandomForestRegressor (rfr.yml) {%=/examples/rfr.yml%}

There are nothing difference to start a run.

```python
run = client.create_run('rfr')
run.start()
```

Because `RandomForestRegressor` estimator has a `criterion` attribute, the metrics are automatically calculated. Take a look at the outputs.

```python
plot(run)
```

### Ridge

#File A parameter YAML file for Ridge (ridge.yml) {%=/examples/ridge.yml%}

Because `Ridge` estimator has no `criterion` attribute, you have to specify metrics if you need. A `mse` key has empty (`None`) value. In this case, the default function (`sklearn.metrics.mean_squared_error()`) is chosen. On the other hand, `mse_2`'s value is a custom function's name:

{{ import rectangle.metrics }}

#Code rectangle.metrics.mean_squared_error() {{ rectangle.metrics.mean_squared_error # inspect }}

This functionality allows us to add arbitrary metrics as long as they can be calculated with `true` and `pred` arrays .

```python
run = client.create_run('ridge')
run.start()  # Both metrics would give the same values.
```

```python
plot(run)
```

## LightGBM

For [LightGBM](https://lightgbm.readthedocs.io/en/latest/), Ivory implements two estimators:

* `ivory.lightgbm.estimator.Regressor`
* `ivory.lightgbm.estimator.Classifier`

#File A parameter YAML file for LightGBM (lgb.yml) {%=/examples/lgb.yml%}

```python
run = client.create_run('lgb')
run.start()
```

```python
plot(run)
```
