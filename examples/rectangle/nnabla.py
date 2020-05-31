import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

import ivory.nnabla.model


class Model(ivory.nnabla.model.Model):
    def __init__(self, hidden_sizes):
        super().__init__()
        self.hidden_sizes = hidden_sizes

    def forward(self, x):
        for k, hidden_size in enumerate(self.hidden_sizes):
            with nn.parameter_scope(f"layer{k}"):
                x = F.relu(PF.affine(x, hidden_size))
        with nn.parameter_scope(f"layer{k+1}"):
            x = PF.affine(x, 1)
        return x
