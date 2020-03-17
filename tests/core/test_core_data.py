def test_data(experiment, run):
    data = experiment.data
    dataloader = run.dataloader
    dataloader.train_ratio = 0.8
    dataloader(data)
    assert len(dataloader.train_dataloader) == 64
    dataloader.val_ratio = 0.5
    dataloader(data)
    assert len(dataloader.val_dataloader) == 10
