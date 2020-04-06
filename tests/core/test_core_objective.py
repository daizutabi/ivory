def test_objective(objective):
    for name in ["suggest_lr", "suggest_hidden_sizes"]:
        assert name in objective.suggests
        assert callable(objective.suggests[name])
