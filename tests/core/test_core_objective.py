from ivory.core.objective import Objective
from ivory.utils.range import Range


def test_objective(objective):
    for name in ["lr", "hidden_sizes"]:
        assert name in objective.suggests
        assert callable(objective.suggests[name])

    assert repr(objective).startswith("Objective(suggests=['lr',")


def test_create_suggest():
    objective = Objective()
    params = {"a": Range(1, 4)}
    objective.create_suggest(params)
    assert "a" in objective.suggests

    params = {"b": Range(0.1, 0.3)}
    objective.create_suggest(params)
    assert "b" in objective.suggests

    params = {"c": Range(0.1, 0.3, num=10), "d": ["a", "b", "c"]}
    objective.create_suggest(params)
    assert "c.d" in objective.suggests
