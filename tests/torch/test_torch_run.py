from ivory.torch.run import dump


def test_dump(run):
    checkpoint = dump(run)
    assert "model" in checkpoint
    assert "optimizer" in checkpoint
    assert "scheduler" in checkpoint
    assert "trainer" in checkpoint
    assert "params" in checkpoint
