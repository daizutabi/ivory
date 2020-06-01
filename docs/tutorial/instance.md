# Creating Instances

{{ ## cache:clear }}

In this tutorial, we will learn about Ivory's internal instance creation system. This is worth to understand the way of writing a YAML file for machine learning.

## Basic idea

A syntax to create an instance is similar to a dictionary.

~~~python
example = ExampleCalss(arg1=123, arg2='abc')
~~~

can be equivalently written as

~~~python
{'example': {'class': 'ExampleCalss', 'args1': 123, 'arg2': 'abc'}}
~~~

Ivory exactly uses this relationship.

```python
from ivory.core.instance import create_instance

params = {'data': {'class': 'rectangle.data.Data', 'n_splits': 5}}
data = create_instance(params, 'data')
data
```

Here, the `create_instance()` requires the second argument `name` to specify a key because the first argument `params` can have multiple keys. Note that we added a `n_splits` parameter that is different from the default value 5. Let's see unique values of fold.

```python
import numpy as np

np.unique(data.fold)  # 5-fold for train and 1-fold for test.
```

For writing a dictionary easily, we use [PyYAML library](https://pyyaml.org/wiki/PyYAMLDocumentation) in this tutorial.

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

Next create a `Dataset` instance. The `Dataset` class requires a `Data` instance as the first argument so that the corresponding dictionary have a hierarchal structure.

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

As you can see, Ivory can treat this hierarchal structure correctly. Next, create a `Datasets` instance.

```python
doc = """
datasets:
  class: ivory.core.data.Datasets
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
    def: ivory.core.data.Dataset
  fold: 0
"""
create(doc, 'datasets')
```

Remember that the argument `dataset` for the `Datasets` class is not an instance but a callable that returns a `Dataset` instance (See [the previous section](../data#datasets)). To describe this behavior, we use a new `def` key to create a callable instead of a `class` key.

## Default Class

In the above example, the two lines using an Ivory's original class seems to be verbose a little bit. Ivory adds a default class if the `class` or `def` key is missing.
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
import torch.utils.data

doc = """
library: torch  # Use default class for PyTorch.
datasets:
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
  fold: 0
"""
datasets  = create(doc, 'datasets')
isinstance(datasets.train, torch.utils.data.Dataset)
```

## Default Value

If a callable has arguments with default value, you can use `__default__` to get the default value from the callable signature.

```python
doc = """
datasets:
  data:
    class: rectangle.data.Data
    n_splits: __default__
  dataset:
  fold: 0
"""
datasets = create(doc, 'datasets')
datasets.data.n_splits
```

## Positional Arguments

Do you know the name of the first argument of `numpy.array()`?

```python
import numpy as np

print(np.array.__doc__[:200])
```

It's `object`. But do you want to write like this?

```python
doc = """
x:
  class: numpy.array  # Or `call` instead of `class`.
  object: [1, 2, 3]
"""
create(doc, 'x')
```

This is inconvenient and ugly. Use **underscore-notation**:


```python
doc = """
x:
  class: numpy.array
  _: [1, 2, 3]
"""
create(doc, 'x')
```

The second argument of `numpy.array()` is `dtype`. You can also use **double underscore**, which is unpacked.

```python
doc = """
x:
  call: numpy.array
  __: [[1, 2, 3], 'float']
"""
create(doc, 'x')
```
