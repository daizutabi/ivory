class Pruner:
    def on_epoch_begin(self, run):
        if run.trainer.epoch == 3:
            run.tracking.client.set_terminated(run.id, "KILLED")


def test_pruning(client):
    run = client.create_run("example")
    run.set(pruner=Pruner())
    run.start()
    trainer = client.load_instance(run.id, "trainer", 'current')
    assert trainer.epoch == 3
