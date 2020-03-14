from ivory.callbacks.metrics import Metrics


def test_record_dict():
    metrics = Metrics()
    assert metrics.record_dict(None) == {}
