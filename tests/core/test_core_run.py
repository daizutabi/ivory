import os


def test_create_run(run):
    assert run.name == "Run#001"


def test_start(run):
    run.start()
    assert run.id
    run.start("test")
    assert run.dataloaders.test


def test_state_dict(run):
    state_dict = run.state_dict()
    assert isinstance(state_dict, dict)
    assert "model" in state_dict
    assert "trainer" in state_dict
    assert "run" not in state_dict

    epoch = run.trainer.epoch
    run.trainer.epoch = epoch + 10
    run.load_state_dict(state_dict)
    assert run.trainer.epoch == epoch


def test_save_and_load(run, tmpdir):
    run.save(tmpdir)
    listdir = os.listdir(tmpdir)
    assert "model" in listdir
    assert "results" in listdir

    epoch = run.trainer.epoch
    state_dict = run.load(tmpdir)
    state_dict["trainer"]["epoch"] == epoch
