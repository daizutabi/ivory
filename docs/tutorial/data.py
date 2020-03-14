# # Data

# All of machine learning need data. So, we will start from data.

# ## Create data

# As a simple toy example, we try to predict `area` of rectangles which have `width` and
# `height`, but they include some noises. First, import necessary libraries.

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch.utils.data
import yaml

import ivory
from ivory.torch import DataLoaders, Dataset
from ivory.utils import kfold_split


# Define a function to create such data.
def create_data(num_samples=100):
    """Returns a tuple of (input, target). Target has fold information."""
    x = 4 * np.random.rand(num_samples, 2) + 1
    x = x.astype(np.float32)
    noises = 0.1 * (np.random.rand(2, num_samples) - 0.5)
    df = pd.DataFrame(x, columns=["width", "height"])
    df["area"] = (df.width + noises[0]) * (df.height + noises[1])
    df.area = df.area.astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df[["width", "height"]], df[["fold", "area"]]


# `kfold_split` function creates a fold-array.

kfold_split(np.arange(10), n_splits=3)

# To execute `create_data` funtion, Ivory provides a simple but powerful method: Use a
# dictionary as a **expression**.

params: Dict[str, Any] = {"data": {"call": "create_data"}}

# This dictionary means: Call a `create_data` function and store the returned value to
# `data`. Apply this dictionary to `ivory.instantiate` function to get an instantiated
# object dictionary.

objects = ivory.instantiate(params)
type(objects), objects.keys()
# -
data = objects["data"]
data[0].head()
# -
data[1].head()

# You can change the default data size with an additional key-value pair.

params = {"data": {"call": "create_data", "num_samples": 5}}
ivory.instantiate(params)["data"][1]

# ## Dataset

# Ivory provides `ivory.torch.Dataset` for PyTorch.

dataset = Dataset(input=data[0], target=data[1])
dataset

# Ivory's `Dataset` is a subclass of PyTorch's `Dataset`


isinstance(dataset, torch.utils.data.Dataset)

# Check an item.

dataset[0]

# Indexing returns a tuple. The first element is an index, the second is an input, and
# the last is a target. The target includes fold which is not a real target. But, it's
# okay because we don't use a `Dataset` directly. We can use more useful `DataLoaders`
# provided by Ivory.

# ## DataLoaders

# `DataLoaders` provides a data loader for both training and validation. This is the
# reason why our data include fold.


dataloaders = DataLoaders(input=data[0], target=data[1], batch_size=3)
dataloaders

# `DataLoaders` detects the number of fold and remove the fold from the target. You can
# get a pair of train and validation data loaders at once by indexing like a `list`.

train_loader, val_loader = dataloaders[0]  # for fold-0.

# Here, the index is corresponding to a fold, in this case, ranging from 0 to 4 because
# the number of K-fold is 5. Check the data loader.

train_loader

# The data loader is a pure PyTorch's `DataLoader`. Let's see the dataset that the data
# loader has.

train_loader.dataset

# This is our Ivory's dataset (`ivory.torch.Dataset`). The number of samples decreases
# from 100 to 80 because we use 5-fold splitting. Validation dataset shoud have the rest
# 20% samples.

val_loader.dataset

# Now check iteration.

next(iter(train_loader))

# `next()` returns a list of `torch.Tensor`. The first element is an index, the second
# is an input, and the last is a *real* target. Note that the target doesn't include
# fold any more. In default setting, a train data loader shuffles its data. A validation
# data loader doesn't:


next(iter(val_loader))

# ## DataLoaders via a YAML file

# You can create your `DataLoaders` from a YAML file instead of a dictionary.

# #File params_1a.yaml {%=params_1a.yaml%}

# Like a function call, if a key is `class`, it means that you need an instance of the
# class. To create a `DataLoaders`, you need to give `input` and `target` to the
# `DataLoaders` initializer. Unlike a simple parameter (`num_samples` or `batch_size` in
# this case), these can't be written in a YAML file directly. Instead, you can assign
# them using "**`$`-notation**". In a YAML file, a value that starts with '`$.`' means
# an instance assigned above. In additon, if the value ends with '`.(digit)`', an
# element is taken from a sequence by indexing. In the above case, '`$.data.0`' is the
# first element of a tuple `data` created by the `create_data` function. Forthermore,
# you can use more direct form: *inline unpacking*.

# #File params_1b.yaml {%=params_1b.yaml%}

# Now, the output of `create_data` is unpacked to `input` and `target` if they are
# joined with a double underscore (`__`). As you imagine. you can join three or more
# instances if you need. In $-notation, '(dot)+(instance name)' can be omitted if the
# key name is equal to an instance name. Next, instantiate them after loading the YAML
# file:


with open("params_1b.yaml") as f:
    params = yaml.safe_load(f)
params
# -
objects = ivory.instantiate(params)

# You can see three objects were created.

objects.keys()
# -
objects["input"].shape, objects["target"].shape
# -
objects["dataloaders"]

# Everythings works well! Notice that the global name space of Python isn't affected by
# these instantiatitions.

try:
    target  # type:ignore
except NameError:
    print("`target` doesn't exsist.")
# In the next section, we will introduce model/optimizer/schduler combination for this
# dataset.
