def test_experiment_run_str(task):
    for k, run in enumerate(task.product(["fold=1-2"], max_epochs="3,4")):
        assert run.dataloaders.fold == [1, 1, 2, 2][k]
        assert run.trainer.max_epochs == [3, 4, 3, 4][k]
