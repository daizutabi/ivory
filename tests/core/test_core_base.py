from ivory.core.base import Base, CallbackCaller


def test_base():
    base = Base({"id": "0", "name": "test", "source_name": "path.yaml"})
    assert not base.objects

    base = Base({}, id="0", name="abc", source_name="path.yaml", a=1)
    assert base.dict
    assert base.id == "0"
    assert base.name == "abc"
    assert base.source_name == "path.yaml"
    assert "num_objects=1" in repr(base)
    assert "a" in base
    assert list(base) == ["a"]
    assert base["a"] == 1


class Callback:
    def on_fit_start(self, caller):
        caller["called"] = True


def test_callback_caller():
    callback = Callback()
    caller = CallbackCaller({}, callback=callback)
    caller.create_callback()
    assert caller.on_fit_start
    assert caller.on_fit_end
    caller.on_fit_end()
    assert not caller.called
    caller.on_fit_start()
    assert caller.called
