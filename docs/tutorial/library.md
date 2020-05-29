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

In this section we compare two libraries and show that using different libraries on the same problem is very easy.

First define models:

#File A Model definition in PyTorch (rectangle/torch.py) {%=/examples/rectangle/torch.py%}

#File A Model definition in TensorFlow (rectangle/tf.py) {%=/examples/rectangle/tf.py%}

For simplicity, the TensorFlow model is defined by using the `keras.Sequential()`, so that we call the `create_model()` function to get the model.

Next, parameter YAML files:

#File A parameter YAML for PyTorch (torch.yml) {%=/examples/torch.yml%}

#File A parameter YAML for TensorFlow (tf.yml) {%=/examples/tf.yml%}

Two YAML files are very similar. Note that `model` for TensorFlow is a function. A new `call` key is used. (But you can use `class`, too, or `call` for a class instead of `class`.)

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

### RandomForestRegressor

#File A parameter YAML for RandomForestRegressor (rfr.yml) {%=/examples/rfr.yml%}

Here, `dataset` has the `transform` argument. This function reshapes the target array to match the shape for scikit-learn estimators.

```python hide
import rectangle.data
```

#Code rectangle.data.transform() {{ rectangle.data.transform # inspect }}

```python
run = client.create_run('rfr')
run.start()
```

Here the RandomForestRegressor estimator has a `criterion` attributes, so the metrics are automatically calculated.

```python
plot(run)
```

### Ridge

#File A parameter YAML for Ridge (ridge.yml) {%=/examples/ridge.yml%}

Because the Ridge estimator has no `criterion` attributes, you have to specify metrics if you need. A `mse` entry has empty (`None`) value. In this case, the default function (`sklearn.metrics.mean_squared_error()`) is chosen. On the other hand, `mse_2`'s value is a custom funtion:

```python hide
import rectangle.metrics
```

#Code rectangle.metrics.mean_squared_error() {{ rectangle.metrics.mean_squared_error # inspect }}

Because the `rectangle.metrics.mean_squared_error()` is the same as `mse`, this example is meaningless, but this functinality allows us to add arbitrary metrics.

```python
run = client.create_run('ridge')
run.start()
```

```python
plot(run)
```

## LightGBM

#File A parameter YAML for LightGBM (lgb.yml) {%=/examples/lgb.yml%}

```python
run = client.create_run('lgb')
run.start()
```

```python
plot(run)
```
