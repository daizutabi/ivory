from ivory.core.objective import create_suggest
from ivory.utils.range import Range


def test_objective(objective):
    for name in ["lr", "hidden_sizes"]:
        assert name in objective.suggests
        assert callable(objective.suggests[name])

    assert repr(objective).startswith("Objective(['lr',")


def test_create_suggest():
    params = {"a": Range(1, 4)}
    suggest = create_suggest(params)
    assert callable(suggest)

    params = {"b": Range(0.1, 0.3)}
    suggest = create_suggest(params)
    assert callable(suggest)

    params = {"c": Range(0.1, 0.3, num=10), "d": ["a", "b", "c"]}
    suggest = create_suggest(params)
    assert callable(suggest)
