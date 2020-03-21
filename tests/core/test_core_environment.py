from ivory.core.environment import create_environment


def test_create_envrionment(params_path):
    env = create_environment(params_path)
    assert len(env) == 2
    assert env.name == ""
