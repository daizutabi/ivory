from ivory.core.run import create_run


def test_create_run(params_path):
    run = create_run(params_path)
    assert run.name == ""


def test_start(run):
    pass
    # run.start()
    # assert run.id
