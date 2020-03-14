import torch

from ivory.torch.utils import cpu, cuda


def test_cpu():
    x = torch.tensor([1, 2])
    if torch.cuda.is_available():
        x = x.cuda()
        assert x.device.type == "cuda"
    y = cpu(x)
    assert y.device.type == "cpu"
    y = cpu([x, x])
    assert y[0].device.type == "cpu"
    assert y[1].device.type == "cpu"


def test_cuda():
    if not torch.cuda.is_available():
        return
    x = torch.tensor([1, 2])
    y = cuda(x)
    assert y.device.type == "cuda"
    y = cuda([x, x])
    assert y[0].device.type == "cuda"
    assert y[1].device.type == "cuda"
