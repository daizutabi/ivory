# # Client, Experiment, and Run

# A `Client` creates and manages an experiment and runs of the experiment.

import os

import ivory
from ivory.core.client import create_client

root = os.path.dirname(ivory.__file__)
path = os.path.join(root, "../tests/params.yaml")
client = create_client(path)
client

# A client instance is a dict-like object. Let's see its contents.
for key, value in client.items():
    print(f"{key}={repr(value)}")

# A client has an automatically created experiment. The experiment is also a dict-like
# object. Let's see its values.

for key, value in client.experiment.items():
    print(f"{key}={repr(value)}")

# `tracker` and `tuner` are copied from the client. The experiment provides data to runs
# that will be created by the client. Let's check the source.

import inspect  # isort:skip

data = client.experiment.data
print(inspect.getsource(data.__class__))

# You can find that the `init` function initializes its data and `get` function returns
# a subset of the data.

data.init()
data.input[:4]

# Now, create a run.

run = client.create_run()
run

# Again, a run is a dict-like object.

for key, value in run.items():
    print(f"{key}={repr(value)}")

# To start the run, just call `start` method.

run.start()

# Check the contents of the run again.

for key, value in run.items():
    print(f"{key}={repr(value)}")

# `on_XXX_begin` and `on_XXX_end` methods were dynamically added. They are callback
# functions invoked by the started run instance. The `repr` of them shows that which
# objects will be called in order.
