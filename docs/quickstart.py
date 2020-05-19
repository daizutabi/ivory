"""md
# Quickstart

## Installing Ivory

You install Ivory by pip command.

~~~bash terminal
$ pip install ivory
~~~
"""

# ## Using the Ivory Client

# Ivory has the `Client` class that manages the workflow of any machine learning. Let's
# create your first `Client` instance. In this quickstart, we are working with examples
# under the `examples` directory.


import numpy as np

import ivory

client = ivory.create_client("examples")
client

# The representation of the `client` shows that it has two objects. Objects that a
# client has can be accessed by **index notation** or **dot notation**.

client[0]  # or client['tracker'], or client.tracker

# The first object is a `Tracker` instance which connects Ivory to [MLFlow
# Tracking](https://mlflow.org/docs/latest/tracking.html).

# You can get all of the objects. Because a `Client` insctance is an iterable, you can
# apply `list` to get the objects.

list(client)

# The second objects is named 'tuner'.

client.tuner

"""md
A `Tuner` instance connects Ivory to [Optuna: A hyperparameter optimization
framework](https://preferred.jp/en/projects/optuna/)

We can customize these objects with a YAML file named `client.yml` under the woking
directory.  In our case, the file just contains the minimum settings.

#File client.yml {%=examples/client.yml%}

!!! note
    A YAML file for client is not required. If there is no file for client, Ivory
    creates a default client with a tracker and without a tuner.

    If you don't need a tracker, use `ivory.create_client(tracker=False)`.


## Example
### Creating NumPy data

In this quickstart, we try to predict rectangles area from thier width and height
using [PyTorch](https://pytorch.org/). First, prepare the data as
[NumPy](https://numpy.org/) arrays. In `example.py` under the working directory, a
`create_data()` function is defined. The `ivory.create_client()` function
automatically inserts the working directory, so that we can import the module.
"""

import example  # isort:skip

"""md
Let's check the `create_data()` function definition and an example output:

#Code example.create_data {{ example.create_data # inspect }}
"""

xy, z = example.create_data(4)
xy
# -
z

"""md
### Set of Data classes

Ivory defines a set of Data classes (`Data`, `Dataset`, `Datasets`, `DataLoaders`).
First one is `ivory.core.data.Data`. Our own `Data` class inherits it.

#Code example.Data {{ example.Data # inspect }}

Here, `kfold_split` function creates a fold-array. In Ivory, fold number = `-1` means
their samples are the test set.
"""
from ivory.utils.fold import kfold_split  # isort:skip

kfold_split(np.arange(10), n_splits=3)

# Now, we can get a `Data` instance.

data = example.Data()
data

"""md
Second class `Dataset` is a class for one fold dataset. We can use a default `Dataset`
for this simple example.
"""
import ivory.core.data  # isort:skip

dataset = ivory.core.data.Dataset(data, mode="train", fold=1)
dataset

# Using `get(index)` method, you can get a list of (index, input, target).

dataset.get(0)

# Next, `Datasets` is a collection class which has `train`, `val`, and `test` dataset.

datasets = ivory.core.data.Datasets(data, ivory.core.data.Dataset, fold=0)
for mode in datasets:
    print(datasets[mode])

# Finally, `DataLoaders` is prepared for PyTorch.

import ivory.torch.data  # isort:skip

dataloaders = ivory.torch.data.DataLoaders(
    data, ivory.torch.data.Dataset, fold=0, batch_size=16
)
for mode in dataloaders:
    print(mode, dataloaders[mode])

"""md
### Defining data by a YAML file

One of the Ivory features is to define everything in a YAML file.

#File data.yaml {%=examples/data.yml%}

Each dictionary value needs to have one of `class`, `call`, `def` key to create an
object. If they are not found, Ivory uses the default classes according to the
dictionary key and `library` value (`torch` in this case). Therfore, `dataloaders` and
`dataset` are created by the default class `ivory.torch.data.DataLoaders` and
`ivory.torch.data.Dataset`.

### Defining a model

We use a simple MLP model here.

#Code example.Model {{ example.Model # inspect }}

### Defining and training a run

Ivory defines a run by a YAML file. Here is a full example.

#File torch.yaml {%=examples/torch.yml%}

Let's creata a run by `client.create_run`
"""

run = client.create_run('torch')
run

# Once you get a run instance, then all you need is to start it.

run.start()

# You can get a history of metrics

run.metrics.history

# Also model output and target.

run.results.val.output[:5]
# -
run.results.val.target[:5]
