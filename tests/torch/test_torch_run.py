def test_dump(run):
    checkpoint = run.dump()
    assert 'model' in checkpoint
    assert 'optimizer' in checkpoint
    assert 'scheduler' in checkpoint
    assert 'trainer' in checkpoint
    assert 'config' in checkpoint
