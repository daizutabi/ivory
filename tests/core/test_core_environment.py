import ivory


def test_create_envrionment(params_path):
    env = ivory.create_environment(params_path)
    assert len(env) == 2
    assert env.name == ""
