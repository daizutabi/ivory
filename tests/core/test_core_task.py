def test_task_product(client, task):
    for k, run in enumerate(task.product({"fold": [1, 2], "epochs": [3, 4]})):
        assert run.dataloaders.fold == [1, 1, 2, 2][k]
        assert run.trainer.epochs == [3, 4, 3, 4][k]

    f = client.search_run_ids
    assert len(list(f("example", parent_run_id=task.id))) == 4
    assert len(list(f("example", parent_run_id=task.id, fold=1))) == 2
    assert run.id not in list(client.search_run_ids(parent_only=True))


def test_task_chain(client, task):
    chain = task.chain({"lr": [2e-3, 1e-4], "epochs": [5, 20]}, use_best_param=False)
    for k, run in enumerate(chain):
        assert run.trainer.epochs == [10, 10, 5, 20][k]
        lr = run.params["run"]["optimizer"]["lr"]
        assert lr == [2e-3, 1e-4, 1e-3, 1e-3][k]

    task = client.create_task('example')
    chain = task.chain({"lr": [1e-13, 1e-4], "epochs": [5, 20], "fold": [2]}, epochs=20)
    for k, run in enumerate(chain):
        assert run.dataloaders.fold == 2
        assert run.trainer.epochs == [20, 20, 5, 20][k]
        lr = run.params["run"]["optimizer"]["lr"]
        assert lr == [1e-13, 1e-4, 1e-4, 1e-4][k]
        run.start()
