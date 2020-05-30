# Library Comparison

{{ ## cache:clear }}

So far, we have used PyTorch in this tutorial, but Ivory can perform machine learning with other libraries.

```python hide
import os
import shutil

if os.path.exists('examples/mlruns'):
    shutil.rmtree('examples/mlruns')
```

## PyTorch vs TensorFlow

In this section we compare two libraries and show that using different libraries on the same problem is straightforward.

First define models:

#File A Model definition in PyTorch (rectangle/torch.py) {%=/examples/rectangle/torch.py%}

#File A Model definition in TensorFlow (rectangle/tf.py) {%=/examples/rectangle/tf.py%}

For simplicity, the TensorFlow model is defined by using the `keras.Sequential()`, so that we call the `create_model()` function to get the model.

Next, write parameter YAML files:

#File A parameter YAML for PyTorch (torch.yml) {%=/examples/torch.yml%}

#File A parameter YAML for TensorFlow (tf.yml) {%=/examples/tf.yml%}

These YAML files have a very similar structure. The first difference comes from that, in PyTorch, an optimizer needs model parameters at the time of instantiation and a scheduler needs an optimizer too, while, in TensorFlow, an optimizer and scheduler can be instantiated without other related instances.  The second difference is `loss` functions. The Pytorch YAML file writes the fullname, while the TensorFlow one writes an abbreviation.


!!! note
    The `model` for TensorFlow is a function. A new `call` key is used. (But you can use `class`, too, or `call` for a class, vice versa, because both a class and function are *callable*.)

Next, create two runs.

```python
import ivory

client = ivory.create_client("examples")
run_torch = client.create_run('torch')
run_tf = client.create_run('tf')
```

For comparison, equalize initial parameters.

```python
import torch

for w_tf, w_torch in zip(run_tf.model.weights, run_torch.model.parameters()):
    w_torch.data = torch.tensor(w_tf.numpy().T)
```

Then, start the runs.

```python
run_torch.start()
```

```python
run_tf.start()
```

Visualize the results:

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

for run in [run_tf, run_torch]:
    plot(run)
```

Actual outputs are like below:

```python
x = run_tf.datasets.test[:10][1]
run_tf.model.predict(x)
```

```python
run_torch.model(torch.tensor(x))
```

## Scikit-learn

Ivory can optimize various scikit-learn's estimators. Here are som examples.

### RandomForestRegressor

#File A parameter YAML for RandomForestRegressor (rfr.yml) {%=/examples/rfr.yml%}

The `dataset` has a `transform` argument. This function reshapes the target array to match the shape for scikit-learn estimators (1D from 2D).

{{ import rectangle.data }}

#Code rectangle.data.transform() {{ rectangle.data.transform # inspect }}

There are nothing different to start a run.

```python
run = client.create_run('rfr')
run.start()
```

Because the RandomForestRegressor estimator has a `criterion` attribute, the metrics are automatically calculated. Take a look at the outputs.

```python
plot(run)
```

### Ridge

#File A parameter YAML for Ridge (ridge.yml) {%=/examples/ridge.yml%}

Because the Ridge estimator has no `criterion` attribute, you have to specify metrics if you need. A `mse` key has empty (`None`) value. In this case, the default function (`sklearn.metrics.mean_squared_error()`) is chosen. On the other hand, `mse_2`'s value is a custom funtion's name:

{{ import rectangle.metrics }}

#Code rectangle.metrics.mean_squared_error() {{ rectangle.metrics.mean_squared_error # inspect }}

Although the `rectangle.metrics.mean_squared_error()` is the same as `mse`, this functionality allows us to add arbitrary metrics as long as they can be calculated with `true` and `pred` values .

```python
run = client.create_run('ridge')
run.start()  # Both metrics would give the same values.
```

```python
plot(run)
```

## LightGBM

For LightGBM, Ivory implements two estimators:

* `ivory.lightgbm.estimator.Regressor`
* `ivory.lightgbm.estimator.Classifier`

#File A parameter YAML for LightGBM (lgb.yml) {%=/examples/lgb.yml%}

```python
run = client.create_run('lgb')
run.start()
```

```python
plot(run)
```
