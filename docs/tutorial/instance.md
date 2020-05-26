# Creating Instance

In this tutorial, we will learn about Ivory's internal instance creation system. This is worth to understand the way of writing a YAML file for machine learning.

We creates a `DataLoaders` described in [the previous section](../data).

## Basic idea

A syntax to create an instance is similar to a dictionary.

    example = ExampleCalss(arg1=123, arg2='abc')

can be equivalently written as

    {'example': {'class': 'ExampleCalss', 'args1': 123, 'arg2': 'abc'}}

Ivory excactly uses this relationship. We call this dictionary `params`.

{{ # cache:clear }}

```python
from ivory.core.instance import create_instance

params = {'data': {'class': 'rectangle.data.Data', 'n_splits': 5}}
data = create_instance(params, 'data')
data
```

Here, `create_instance()` requires the second parameter `name` to specify a key because `params` can have multiple keys. Note that we added `n_splits` parameter which is different from the default value 5. Let's see unique values of fold.

```python
import numpy as np

np.unique(data.fold)
```

For writing `parmas` easily, we use [PyYAML library](https://pyyaml.org/wiki/PyYAMLDocumentation).

```python
import yaml

# A helper function.
def create(doc, name, **kwargs):
    params = yaml.safe_load(doc)
    return create_instance(params, name, **kwargs)

doc = """
data:
  class: rectangle.data.Data
  n_splits: 5
"""
create(doc, 'data')
```

## Hierarchal Structure

Next create a `Dataset` instance. The `Dataset` class requires a `Data` instance as the first argument so that the corresponding dictionary have hierarchal structure.

```python
doc = """
dataset:
  class: ivory.core.data.Dataset
  data:
    class: rectangle.data.Data
    n_splits: 5
  mode: train
  fold: 0
"""
create(doc, 'dataset')
```

As you can see, Ivory can treat this hierarchal structure correctly. Furthermore, create a `DataLoaders` instance for PyTorch.

```python
doc = """
dataloaders:
  class: ivory.torch.data.DataLoaders
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
    def: ivory.core.data.Dataset
  fold: 0
  batch_size: 4
"""
create(doc, 'dataloaders')
```

Remember that the argument `dataset` for the `DataLoaders` class is not an instance but a callable that returns a `Dataset` instance. To describe this behavior, you can use a new `def` key instead of `class`. (See [the previous section](../data#dataloaders))

## Default Class

In the above example, the two lines using a class of Ivory seems to be verbose a little bit. Ivory adds a default class if the `class` or `def` key is missing.
Here is the list of default classes prepared by Ivory:

```python
from ivory.core.default import DEFAULT_CLASS

for library, values in DEFAULT_CLASS.items():
    print(f'library: {library}')
    for name, value in values.items():
        print("    ", name, "---", value)
```

Therefore, we can omit the lines using default classes like below. Here, the `library` key is used to overload the default classes of the `ivory.core` package by the specific library.

```python
doc = """
library: torch
dataloaders:
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
  fold: 0
  batch_size: 4
"""
create(doc, 'dataloaders')
```
