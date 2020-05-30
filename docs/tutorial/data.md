# Set of Data classes

{{ ## cache:clear }}

Ivory uses three classes for data presentation: `Data`, `Dataset`, and `Datasets`. In this tutorial, we use the following Python module to explain them.

#File rectangle/data.py {%=/examples/rectangle/data.py%}

## Data Class

First import the module and check the basic behavior.

```python
import rectangle.data

data = rectangle.data.Data()
data
```

In the `Data.init()` method, we need to define 4 attributes:

* `index`: Index of samples.
* `input`: Input data.
* `target`: Target data.
* `fold`: Fold number.

A `Data.get()` method returns a tuple of (`index`, `input`, `target`). This method is called from the `Dataset` instance when the dataset is indexed.

```python
data.get(0)  # Integer index.
```

```python
data.get([0, 10, 20])  # Array-like index. list or np.ndarray
```

## Dataset

An instance of the `Dataset` class holds one of train, validation, and test dataset. We use the Ivory's default `Dataset` class here instead of defining a subclass. The `Dataset()` initializer requires three arguments: A `Data` instance, `mode`, and `fold`.

```python
import ivory.core.data

dataset = ivory.core.data.Dataset(data, 'train', 0)
dataset
```

```python
ivory.core.data.Dataset(data, 'val', 1)  # Another mode is `test`.
```

As the `Data` class, the `Dataset` class has a `init()` method without any arguments and no returned value.  You can define any code to modify data.

To get data from an dataset. use normal indexing

```python
dataset[0]  # Integer index.
```

```python
dataset[[0, 10, 20]]  # Array-like index. list or np.ndarray
```
```python
index, *_ = dataset[:]  # Get all data.
print(len(index))
index[:10]
```

These data come from a subset of the `data` instance according to the mode and fold.

The `Dataset` class takes an opptional and callable argument: `transform`.

```python
def transform(mode: str, input, target):
    if mode == 'train':
        input = input * 2
        target = target * 2
    return input, target

dataset_transformed = ivory.core.data.Dataset(data, 'train', 0, transform)
dataset_transformed[0]
```

```python
2 * dataset[0][1], 2 * dataset[0][2]
```

Usually, we don't instantiate the `Dataset` class directly. Instead, the `Datasets` class create dataset instances.

## Datasets

An instance of the `Datasets` class holds a set of train, validation, and test dataset. We use the Ivory's default `Datasets` class here instead of defining a subclass. The `Datasets()` initializer requires three arguments: A `Data` instance, `Dataset` factory, and `fold`.

```python
from ivory.core.data import Dataset

datasets = ivory.core.data.Datasets(data, Dataset, 0)
datasets
```

!!! note
    The second argument (`dataset`) is not a `Dataset` instance but its factory that returns a `Dataset` instance. It may be a `Dataset` class itself or any function that returns a `Dataset` instance.

```python
for mode, dataset in datasets.items():
    print(mode, dataset)
```

Each dataset can be accessed by indexing or attributes.

```python
datasets['train'], datasets.val
```

Using the `Datasets` class, we can easily split a whole data stored in a `Data` instance into three train, validation, and test dataset.
