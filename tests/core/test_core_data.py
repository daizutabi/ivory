def test_sample(dataloaders):
    dataset = dataloaders.train.dataset
    index = dataset.get()[0]
    n = len(index)
    index = dataset.sample(10)[0]
    assert len(index) == 10
    index = dataset.sample(frac=0.1)[0]
    assert len(index) == int(0.1 * n)
