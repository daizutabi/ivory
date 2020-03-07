from ivory.core.runner import create_runner
from ivory.core.instance import instantiate


def test_runner(config):
    config.append({'runner': {'class': 'ivory.core.runner.Runner'}})
    runner = create_runner(config)
    assert all(runner.data == [1, 2])

    default = instantiate(config, keys=['data'])
    runner1 = create_runner(config)
    runner2 = create_runner(config)
    assert runner1.data is not runner2.data
    runner1 = create_runner(config, default=default)
    runner2 = create_runner(config, default=default)
    assert runner1.data is runner2.data
    assert runner1.data is default.data
