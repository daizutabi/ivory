import numpy as np
import pytest

from ivory.sklearn.estimator import RandomForestClassifier, Ridge


def test_rfr(client):
    experiment = client.create_experiment("rfr")
    run = experiment.create_run()
    run.start()
    assert run.results.train["output"].shape == (600,)


def test_rfc():
    rfc = RandomForestClassifier(n_estimators=10)
    x = np.random.rand(100, 20)
    y = np.random.randint(0, 10, 100)
    rfc.fit(x, y)
    assert rfc.predict(x).shape == (100, 10)

    rfc.return_probability = False
    assert rfc.predict(x).shape == (100,)


def test_unknown_kwargs():
    rfc = RandomForestClassifier(n_estimators=10)
    assert "n_estimators" in rfc.kwargs
    with pytest.raises(ValueError):
        rfc = RandomForestClassifier(abc=10)


def test_repr():
    ridge = Ridge()
    assert repr(ridge) == "Ridge()"
    rfc = RandomForestClassifier(n_estimators=10)
    assert repr(rfc) == "RandomForestClassifier(n_estimators=10)"
