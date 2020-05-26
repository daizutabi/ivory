# Model Structure

## Model

We have prepared a `DataLoaders` instance for PyTorch. Now define a MLP model that works with the `DataLoaders`.

The model is defined in `rectangle/torch.py`

#File rectangle/torch.py {%=/docs/src/rectangle/torch.py%}

We again use Ivory's [instance creation system](../instance).

{{ # cache:clear }}

```python
import yaml

# A helper function.
def create(doc, name, **kwargs):
    params = yaml.safe_load(doc)
    return create_instance(params, name, **kwargs)

doc = """
library: torch
dataloaders:
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
  fold: 0
  batch_size: 4
model:
  class: rectangle.torch.Model
  hidden_sizes: [3, 4, 5]
"""
dataloaders = create(doc, 'dataloaders')
model = create(doc, 'model')
model
```

We can uses this model as usual.

```python
index, input, target = next(iter(dataloaders.train))
input
```

```python
model(input)
```

## Optimizer

To train a model, we need an optimizer. For example

```python
import torch.optim

optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
optimizer
```

Now try to describe this optimizer in a dictionary style. However, the argument `params` is not a simple literal but an iterable of learnable parameters. Ivory provides "**`$`-notation**" to tackle this problem.


```python
doc = """
optimizer:
  class: torch.optim.SGD
  params: $.model.parameters()
  lr: 0.001
"""
optimizer = create(doc, 'optimizer', globals={'model': model})
optimizer
```

A "**`$`**" is a starting point to refer other instance stored in a `globals` dictionary. In this case, `$.model` is replaced by the `model` instance in `globals`, then `.parameters()` invokes a call of the `model.parameters()` method.


## Scheduler

A Scheduler controls the learning rate of an optimizer.

```python
doc = """
scheduler:
  class: torch.optim.lr_scheduler.ReduceLROnPlateau
  optimizer: $
  factor: 0.5
  patience: 4
"""
scheduler = create(doc, 'scheduler', globals={'optimizer': optimizer})
scheduler
```

If a `$`-notation has no suffix, the value becomes its key itself. The following two example are equivalent:

    optimizer: $
    optimizer: $.optimizer

Now we have had both data and model.
