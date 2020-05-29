import numpy as np
import pytest

from example import Dataset


def test_dataset(data):
    data.init()
    assert data.input.shape == (1000, 2)
    assert data.fold.shape == (1000,)
    dataset = Dataset(data, "train", 0)
    assert len(dataset) == 600
    assert len(list(iter(dataset))) == 600
    assert "num_samples=600" in repr(dataset)
    with pytest.raises(IndexError):
        dataset[800]


def test_dataset_transform(data):
    def transform(mode, input, target):
        return input * 2, target / 2

    dataset = Dataset(data, "train", 1, transform)
    index, input, target = dataset[0]
    assert index == 0
    assert np.allclose(input, data.input[0] * 2)
    assert np.allclose(target, data.target[0] / 2)
