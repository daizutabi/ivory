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
    assert "target_shape=" not in repr(dataset)


def test_dataset_transform(data):
    def transform(input, output, mode):
        return input * 2, output / 2

    dataset = Dataset(data[["x", "y"]], data[["z"]], transform)
    index, input, target = dataset[0]
    assert index == 0
    assert np.allclose(input, data.loc[0, ["x", "y"]] * 2)
    assert np.allclose(target, data.loc[0, ["z"]] / 2)
    assert "transform=" in repr(dataset)


def test_dataloaders(run):
    dataloaders = run.dataloaders
    assert len(dataloaders) == 5
    train_loader, val_loader = dataloaders[0]
    assert len(train_loader) == 1000 * 4 // 5 // 10
    assert len(val_loader) == 1000 * 1 // 5 // 10
    assert train_loader.dataset.mode == "train"
    assert val_loader.dataset.mode == "val"
    dataloaders.train_percent_check = 0.5
    dataloaders.val_percent_check = 0.2
    train_loader, val_loader = dataloaders[0]
    assert len(train_loader) == int(1000 * 4 // 5 // 10 * 0.5)
    assert len(val_loader) == int(1000 * 1 // 5 // 10 * 0.2)

    assert repr(dataloaders).startswith("DataLoaders")
