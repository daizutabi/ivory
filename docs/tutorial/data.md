# Set of Data classes

{{ ## cache:clear }}

Ivory uses four classes for data presentation: `Data`, `Dataset`, `Datasets`, and `DataLoaders`. In this tutorial, we use the following Python module to explain them.

#File rectangle/data.py {%=/examples/rectangle/data.py%}

## Data Class

First import the module and check the basic behavior.

```python
import rectangle.data

data = rectangle.data.Data()
data
```

In `Data.init()`, we need to define 4 attributes:

* `index`: Index of samples.
* `input`: Input data.
* `target`: Target data.
* `fold`: Fold number.

`Data.get()` returns a tuple of (`index`, `input`, `target`). This function is called from `Dataset` instances when the dataset is indexed.

```python
data.get(0)  # Integer index.
```

```python
data.get([0, 10, 20])  # Array-like index. list or np.ndarray
```

## Dataset Class

An instance of the `Dataset` class holds one of train, validation, and test dataset. We use the Ivory's default `Dataset` here instead of defining a subclass. `Dataset()` initializer requires three arguments: A `Data` instance, `mode`, and `fold`.

```python
import ivory.core.data

dataset = ivory.core.data.Dataset(data, 'train', 0)
dataset
```

```python
ivory.core.data.Dataset(data, 'val', 1)  # Another mode is `test`.
```

As the `Data`, the `Dataset` has `init()` without any arguments and returned value.  You can define any code to modify data.

To get data from an `Dataset` instance, use normal indexing

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

These data come from a subset of the `Data` instance according to the mode and fold.

The `Dataset` takes an optional and callable argument: `transform`.

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

Usually, we don't instantiate the `Dataset` directly. Instead, the `Datasets` class create dataset instances.

## Datasets Class

An instance of the `Datasets` class holds a set of train, validation, and test dataset. We use the Ivory's default `Datasets` here instead of defining a subclass. The `Datasets()` initializer requires three arguments: A `Data` instance, `Dataset` factory, and `fold`.

```python
from ivory.core.data import Dataset

datasets = ivory.core.data.Datasets(data, Dataset, 0)
datasets
```

!!! note
    The second argument (`dataset`) is not a `Dataset` instance but its factory that returns a `Dataset` instance. It may be a `Dataset` itself or any other function that returns a `Dataset` instance.

```python
for mode, dataset in datasets.items():
    print(mode, dataset)
```

Each dataset can be accessed by indexing or attributes.

```python
datasets['train'], datasets.val
```

Using the `Datasets`, we can easily split a whole data stored in a `Data` instance into three train, validation, and test dataset.

## DataLoaders Class

The `DataLoaders` class is used internally by `ivory.torch.trainer.Trainer` or
`ivory.nnabla.trainer.Trainer` classes to yield a minibatch in training loop.

```python
from ivory.torch.data import DataLoaders

dataloaders = DataLoaders(datasets, batch_size=4, shuffle=True)
dataloaders
```

```python
for mode, dataloader in dataloaders.items():
    print(mode, dataloader)
```

```python
next(iter(dataloaders.train))  # Shuffled
```

```python
next(iter(dataloaders.val))  # Not shuffled, regardless of `shuffle` argument
```

```python
next(iter(dataloaders.test))  # Not shuffled, regardless of `shuffle` argument
```
