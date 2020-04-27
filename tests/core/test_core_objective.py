def test_objective(objective):
    for name in ["lr", "hidden_sizes"]:
        assert name in objective.suggests
        assert callable(objective.suggests[name])
