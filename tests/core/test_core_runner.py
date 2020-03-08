from ivory.core.run import Run
from ivory.core.instance import instantiate


def test_run(config):
    run = Run(config)
    assert all(run.data == [1, 2])

    default = instantiate(config, names=['data'])
    run1 = Run(config)
    run2 = Run(config)
    assert run1.data is not run2.data
    run1 = Run(config, default=default)
    run2 = Run(config, default=default)
    assert run1.data is run2.data
    assert run1.data is default['data']
