import pytest

from ivory.core.default import update_class


def test_update_class():
    with pytest.raises(ValueError):
        update_class({"abc": None})
