import numpy as np

from example import Dataset
from ivory.core.instance import create_instance


def test_dataset(data):
    data.initialize()
    assert data.input.shape == (1000, 2)
    assert data.fold.shape == (1000,)
    dataset = Dataset("train", data.get())
    assert len(dataset) == 1000
    dataset = Dataset("train", data.get([1, 2, 3]))
    assert len(dataset) == 3
    assert "num_samples=3" in repr(dataset)


def test_dataset_transform(data):
    def transform(mode, input, target):
        return input * 2, target / 2

    dataset = Dataset("train", data.get([0, 2, 3]), transform)
    index, input, target = dataset[0]
    assert index == 0
    assert np.allclose(input, data.input[0] * 2)
    assert np.allclose(target, data.target[0] / 2)


def test_dataloader(dataloader, data):
    dataloader.init(data)
    train_loader, val_loader = dataloader.train, dataloader.val
    assert len(train_loader) == 1000 * 4 // 5 // 10
    assert len(val_loader) == 1000 * 1 // 5 // 10
    assert train_loader.dataset.mode == "train"
    assert val_loader.dataset.mode == "val"


def test_dataloader_test(dataloader, data):
    data.mode = "test"
    data.initialized = False
    dataloader.init(data)
    test_loader = dataloader.test
    assert len(test_loader) == 1000 // 10
    assert test_loader.dataset.mode == "test"


def test_dataloader_repr(dataloader, data, client, params):
    assert "dataset=example.Dataset(dummy=5)" in repr(dataloader)
    params["run"]["dataloader"]["dataset"].pop("dummy")
    dataloader = create_instance(params, "run.dataloader")
    assert "dataset=example.Dataset()" in repr(dataloader)
