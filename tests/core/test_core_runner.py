from ivory.core.runner import Runner
from ivory.core.instance import instantiate


def test_runner(config):
    runner = Runner(config)
    assert all(runner.data == [1, 2])

    default = instantiate(config, names=['data'])
    runner1 = Runner(config)
    runner2 = Runner(config)
    assert runner1.data is not runner2.data
    runner1 = Runner(config, default=default)
    runner2 = Runner(config, default=default)
    assert runner1.data is runner2.data
    assert runner1.data is default['data']
