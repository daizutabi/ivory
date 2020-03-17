from ivory.core.base import Base, CallbackCaller


def test_base():
    base = Base({"id": "0", "name": "test"})
    assert not base.objects

    base = Base({}, id="0", name="abc", a=1)
    assert base.objects
    assert base.id == "0"
    assert base.name == "abc"
    assert "num_objects=1" in repr(base)
    assert "a" in base
    assert list(base) == ["a"]
    assert base["a"] == 1


class Callback:
    def on_fit_start(self, caller):
        caller.objects["called"] = True


def test_callback_caller():
    callback = Callback()
    caller = CallbackCaller({}, callback=callback)
    caller.create_callbacks()
    assert caller.on_fit_start
    assert caller.on_fit_end
    caller.on_fit_end()
    assert not caller.called
    caller.on_fit_start()
    assert caller.called
