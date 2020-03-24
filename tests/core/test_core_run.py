def test_create_run(run):
    assert run.name == ""


def test_start(run):
    run.start()
    assert run.id
