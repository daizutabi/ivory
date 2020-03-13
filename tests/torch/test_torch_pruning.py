def test_pruning(pruning):
    assert "monitor='val_loss'" in repr(pruning)
