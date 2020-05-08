def test_mnist(client):
    run = client.create_run("mnist")
    run.start("train")
    run.start("test")
    assert run.results.train["output"].shape == (48000, 10)
    assert run.results.val["output"].shape == (12000, 10)
    assert run.results.test["output"].shape == (10000, 10)
