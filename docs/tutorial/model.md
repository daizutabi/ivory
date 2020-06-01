# Model Structure

{{ ## cache:clear }}

## Model

We have prepared a `Datasets` instance for PyTorch. Now define a MLP model that works with this `Datasets`.

The model is defined in `rectangle/torch.py`

#File rectangle/torch.py {%=/examples/rectangle/torch.py%}

We again use Ivory's [instance creation system](../instance).

```python
import yaml

# A helper function.
def create(doc, name, **kwargs):
    params = yaml.safe_load(doc)
    return create_instance(params, name, **kwargs)

doc = """
library: torch
datasets:
  data:
    class: rectangle.data.Data
    n_splits: 5
  dataset:
  fold: 0
model:
  class: rectangle.torch.Model
  hidden_sizes: [3, 4, 5]
"""
datasets = create(doc, 'datasets')
model = create(doc, 'model')
model
```

We can uses this model as usual.

```python
import torch

index, input, target = datasets.train[:5]
input
```

```python
model(torch.tensor(input))
```

## Optimizer

To train a model, we need an optimizer. For example

```python
import torch.optim

optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
optimizer
```

Now try to describe this optimizer in a dictionary style. However, the first argument `params` is neigher a simple literal nor an other instance. It is an iterable of learnable parameters obtained from a model. Ivory provides "**`$`-notation**" to tackle this problem.


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

A "**`$`**" is a starting point to refer other instance stored in the `globals` dictionary. In this case, `$.model` is replaced by the `model` instance in `globals`, then `.parameters()` invokes a call of `Model.parameters()`.


## Scheduler

A scheduler controls the learning rate of an optimizer.

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

If a `$`-notation has no suffix, the value becomes its key itself. The following two examples are equivalent:


~~~yaml
optimizer: $
~~~

~~~yaml
optimizer: $.optimizer
~~~

Now we have had both data and a model.
