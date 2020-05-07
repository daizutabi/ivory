def test_load_results(evaluator):
    run = evaluator.client.create_run('example')
    run.start("train")
    run.start("test")
    evaluator.run_ids = [run.id, run.id]
    output, target = evaluator.load_results()
    assert len(output) == 200 * 2 * 2


def test_from_results(evaluator):
    run = evaluator.client.create_run('example')
    run.start("train")
    run.start("test")
    evaluator.run_ids = [run.id, run.id]
    output, target = evaluator.from_results(softmax=True)
    assert len(output) == 200 * 2
