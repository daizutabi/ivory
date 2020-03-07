from ivory import Runner


def test_runner(config):
    runner = Runner.create(config)
    assert all(runner.cfg.data == [1, 2])
