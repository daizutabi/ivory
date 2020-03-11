from ivory.core.instance import instantiate
from ivory.core.run import Run


def test_run(params):
    run = Run(params=params)
    assert len(run) == 5
    assert all(run.data == [1, 2])
    assert "data" in run
    assert run.a == 3
    assert run.b == 4

    default = instantiate({"data": params["data"]})
    run1 = Run(params=params)
    run2 = Run(params=params)
    assert run1.data is not run2.data
    run1 = Run(params=params, default=default)
    run2 = Run(params=params, default=default)
    assert run1.data is run2.data
    assert run1.data is default["data"]

    state_dict = run.state_dict()
    assert state_dict["metrics"]["best_epoch"] == run.metrics.best_epoch
    state_dict["metrics"]["best_epoch"] = 100
    run.load_state_dict(state_dict)
    assert run.metrics.best_epoch == 100
