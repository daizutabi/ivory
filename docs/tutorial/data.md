# Data

## Create data

As a simple toy example, we try to predict `area` of rectangles which have `width` and `height`, but they include some noises.

First, define a function to create such data.

```python
import numpy as np
import pandas as pd

import ivory
from ivory.utils import kfold_split

def create_data(num_samples=1000):
    """Returns a tuple of (input, target). Target has fold information."""
    x = 4 * np.random.rand(num_samples, 2) + 1
    x = x.astype(np.float32)
    noises = 0.1 * (np.random.rand(2, num_samples) - 0.5)
    df = pd.DataFrame(x, columns=["width", "height"])
    df["area"] = (df.width + noises[0]) * (df.height + noises[1])
    df.area = df.area.astype(np.float32)
    df["fold"] = kfold_split(df.index, n_splits=5)
    return df[["width", "height"]], df[["fold", "area"]]
```

`kfold_split` function creates a fold-array.

```python
kfold_split(np.arange(10), n_splits=3)
```

To execute `create_data` funtion, you can use a dictionary as well as a YAML file.

```python
params = {'data': {'def': 'create_data'}}
```

The key of `def` means that the value is a funtion instead of a class. Apply this dictionary to `ivory.instantiate` function to get an instantiated object dictionary.

```python
data = ivory.instantiate(params)['data']
data[0].head()
```

```python
data[1].head()
```

You can change the data size with additional key-value pairs (in this case, the minimun size is the number of fold 5).

```python
params = {'data': {'def': 'create_data', 'num_samples': 5}}
ivory.instantiate(params)['data'][1]
```

## Dataset

Ivory provides `ivory.torch.Dataset` for PyTorch.

```python
from ivory.torch import Dataset

dataset = Dataset(data[0], data[1])
dataset
```

Ivory's `Dataset` is a subclass of PyTorch's `Dataset`

```python
import torch.utils.data

isinstance(dataset, torch.utils.data.Dataset)
```

Check an item.

```python
dataset[0]
```

Indexing returns a tuple. The first is an index, the second is an input, and the last is a target. The target includes fold which is not a real target. But, it's okay because we don't use a `Dataset` directly. We can use more usefull `DataLoaders` provided by Ivory.

## DataLoaders

`DataLoaders` provides a data loader for both training and validation. This is the reason why our data include fold information.

```python
from ivory.torch import DataLoaders

dataloaders = DataLoaders(data[0], data[1], batch_size=3)
dataloaders
```

`DataLoaders` instance can detect the number of fold and remove the fold information from the target. Normal indexing can be used to get a pair of train data loader and validation data loader.

```python
train_loader, val_loader = dataloaders[0]
```

Here, index means fold number, in this case, ranging from 0 to 4 because the number of K-fold is 5. Check the data loarder.

```python
train_loader
```

The data loader is a pure PyTorch's `DataLoader`. Let's see the dataset that the data loader has.


```python
train_loader.dataset
```

This is our Ivory's dataset. The number of samples reduces from 1000 to 800 because we uses 5 folds (80% reduction). Validation dataset shoud have the rest 20% samples.

```python
val_loader.dataset
```

Now check iteration.

```python
it = iter(train_loader)
next(it)
```

`next()` returns a list of `torch.Tensor`. The first is an index, the second is an input, and the last is a target. Note that the target doesn't include fold any more. In default setting, train data loader shuffles its data. Validation data loader doesn't:


```python
it = iter(val_loader)
next(it)
```


## DataLoaders
