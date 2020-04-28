def test_task(client, task):
    for k, run in enumerate(task.product(["fold=1-2"], max_epochs="3,4")):
        assert run.dataloaders.fold == [1, 1, 2, 2][k]
        assert run.trainer.max_epochs == [3, 4, 3, 4][k]
        if k != 0:
            run.start()

    f = client.search_run_ids
    assert len(list(f("example", parent_run_id=task.id))) == 4
    assert len(list(f("example", parent_run_id=task.id, fold=1))) == 1
    assert run.id not in list(client.search_run_ids(parent_only=True))
