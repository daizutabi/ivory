# # Model

# We will train a model to predict a toy rectangle area problem. First, define a simple
# model with `hidden_sizes`, which can be a hyper parameter (list of integer).

import torch.nn as nn
import torch.nn.functional as F
import yaml

import ivory
from ivory.utils import to_float


class Model(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        it = zip([2] + hidden_sizes, hidden_sizes + [1])
        for in_features, out_features in it:
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


Model([3, 4])  # Just example

# To train this model, we need an optimizer and an optional schduler. Write them in a
# YAML file too.

# #File params_2.yaml {%=params_2.yaml%}

# An PyTorch optimizer needs model's parameters as the first argument. We can give them
# by $-notation with attribute accessor (`$.model.parameters()`). The rest part of this
# YAML file is straight forward. Now, we have other hyper parameters such as `lr`
# (learning rate), `factor`, or `patience`. All of these hyper parameters are written in
# one YAML file. This allows us to manage them easily.

# Again, let's create instances:


with open("params_2.yaml") as f:
    params = to_float(yaml.safe_load(f))
objects = ivory.instantiate(params)
objects.keys()

# Here, a helper function `ivory.utils.to_float` converts an exponential expression such
# as "1e-3" to "0.001". Check the created instances

objects["model"]
# -
objects["optimizer"]
# -
objects["scheduler"]

# Now, we have data and a model with an optimizer so we can start to *train*.

dataloader, _ = objects["dataloaders"][0]
index, input_, target = next(iter(dataloader))
input_[:4]
# -
output = objects["model"](input_)
output[:4]

# Calculate the first loss

loss = F.mse_loss(output, target)
loss
# Backpropagate and update the weights.

objects["optimizer"].zero_grad()
loss.backward()
objects["optimizer"].step()

# Calculate the second loss after the first optimization step.

output = objects["model"](input_)
loss = F.mse_loss(output, target)
loss

# To train a model, we need *metrics* to estimate the model performance. In the next
# section, we will introduce `Metrics` class for it.
