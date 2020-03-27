from ivory.callback.metrics import Metrics


def test_metrics_dict():
    metrics = Metrics()
    assert metrics.metrics_dict(None) == {}
