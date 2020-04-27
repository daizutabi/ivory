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


def test_dataloaders(dataloaders, data, dataset):
    train_loader, val_loader = dataloaders.train, dataloaders.val
    assert len(train_loader) == 1000 * 3 // 5 // 10
    assert len(val_loader) == 1000 * 1 // 5 // 10
    assert train_loader.dataset.mode == "train"
    assert val_loader.dataset.mode == "val"


def test_dataloaders_test(dataloaders, data, dataset):
    test_loader = dataloaders.test
    assert len(test_loader) == 1000 * 1 // 5 // 10
    assert test_loader.dataset.mode == "test"
