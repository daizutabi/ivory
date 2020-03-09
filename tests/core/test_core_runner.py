from ivory.core.run import Run
from ivory.core.instance import instantiate


def test_run(params):
    run = Run(params)
    assert all(run.data == [1, 2])

    default = instantiate(params, names=['data'])
    run1 = Run(params)
    run2 = Run(params)
    assert run1.data is not run2.data
    run1 = Run(params, default=default)
    run2 = Run(params, default=default)
    assert run1.data is run2.data
    assert run1.data is default['data']
