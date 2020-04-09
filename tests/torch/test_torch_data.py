import numpy as np
import pytest

from example import Dataset
from ivory.core.instance import create_instance


def test_dataset(data):
    data.init()
    assert data.input.shape == (1000, 2)
    assert data.fold.shape == (1000,)
    dataset = Dataset("train", data.get("train", [1, 2, 3]))
    assert len(dataset) == 3
    assert "num_samples=3" in repr(dataset)
    with pytest.raises(IndexError):
        dataset[4]


def test_dataset_transform(data):
    def transform(mode, input, target):
        return input * 2, target / 2

    dataset = Dataset("train", data.get("train", [0, 2, 3]), transform)
    index, input, target = dataset[0]
    assert index == 0
    assert np.allclose(input, data.input[0] * 2)
    assert np.allclose(target, data.target[0] / 2)


def test_dataloaders(dataloaders, data):
    dataloaders.init("train", data)
    train_loader, val_loader = dataloaders.train, dataloaders.val
    assert len(train_loader) == 1000 * 4 // 5 // 10
    assert len(val_loader) == 1000 * 1 // 5 // 10
    assert train_loader.dataset.mode == "train"
    assert val_loader.dataset.mode == "val"


def test_dataloaders_test(dataloaders, data):
    dataloaders.init("test", data)
    test_loader = dataloaders.test
    assert len(test_loader) == 1000 // 10
    assert test_loader.dataset.mode == "test"


def test_dataloader_repr(dataloaders, data, client, params):
    assert "dataset=example.Dataset(dummy=5)" in repr(dataloaders)
    params["run"]["dataloaders"]["dataset"].pop("dummy")
    dataloaders = create_instance(params, "run.dataloaders")
    assert "dataset=example.Dataset()" in repr(dataloaders)
