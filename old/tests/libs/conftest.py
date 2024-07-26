import os
import shutil
import sys

import pytest
import torch

from ivory.core.client import create_client


@pytest.fixture(scope="module")
def runs():
    sys.path.insert(0, os.path.abspath("examples"))
    client = create_client(directory="examples")
    runs = []
    for name in ["tensorflow", "nnabla", "torch2"]:
        run = client.create_run(name, epochs=5, batch_size=10, shuffle=False)
        runs.append(run)
    run_tf, run_nn, run_torch = runs

    run_nn.model.build(
        run_nn.trainer.loss, run_nn.datasets.train, run_nn.trainer.batch_size
    )
    run_nn.optimizer.set_parameters(run_nn.model.parameters())

    ws_tf = run_tf.model.weights
    ws_nn = run_nn.model.parameters().values()
    ws_torch = run_torch.model.parameters()
    for w_tf, w_nn, w_torch in zip(ws_tf, ws_nn, ws_torch):
        w_nn.data.data = w_tf.numpy()
        w_torch.data = torch.tensor(w_tf.numpy().T)

    yield dict(zip(["tf", "nn", "torch"], runs))
    del sys.path[0]
    if os.path.exists("examples/mlruns"):
        shutil.rmtree("examples/mlruns")
