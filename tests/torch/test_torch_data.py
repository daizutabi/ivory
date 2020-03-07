import numpy as np

from ivory.torch.data import Dataset


def test_dataset(data):
    assert data.shape == (1000, 4)
    input = data[["x", "y"]].to_numpy()
    target = data[["z"]].to_numpy()
    dataset = Dataset(input, target)
    assert len(dataset) == 1000
    assert dataset.index[-1] == 999
    assert len(dataset[0]) == 3
    dataset = Dataset(input)  # without target
    assert len(dataset[0]) == 2
    assert "target_shape=None" in repr(dataset)


def test_dataset_from_dataframe(data):
    dataset = Dataset.from_dataframe(data, ["x", "y"], ["z"])
    assert len(dataset) == 1000
    assert repr(dataset).startswith("Dataset")


def test_dataset_transform(data):
    def transform(input, output):
        return input * 2, output / 2

    dataset = Dataset.from_dataframe(data, ["x", "y"], ["z"], transform)
    index, input, target = dataset[0]
    assert index == 0
    assert np.allclose(input, data.loc[0, ["x", "y"]] * 2)
    assert np.allclose(target, data.loc[0, ["z"]] / 2)


def test_dataloaders(dataloaders):
    assert len(dataloaders) == 5
    train_loaders, val_loaders = dataloaders[0]
    assert len(train_loaders) == 1000 * 4 // 5 // 10
    assert len(val_loaders) == 1000 * 1 // 5 // 10
    dataloaders.train_percent_check = 0.5
    dataloaders.val_percent_check = 0.2
    train_loaders, val_loaders = dataloaders[0]
    assert len(train_loaders) == int(1000 * 4 // 5 // 10 * 0.5)
    assert len(val_loaders) == int(1000 * 1 // 5 // 10 * 0.2)

    assert repr(dataloaders).startswith("DataFrameLoaders")
