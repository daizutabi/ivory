import numpy as np

from ivory.callbacks.results import concatenate


def test_libraries(runs):
    for run in runs.values():
        run.start("both")

    for mode in ["val", "test"]:
        outputs = []
        for run in runs.values():
            outputs.append(run.results[mode].output)

        for output in outputs[1:]:
            assert np.allclose(outputs[0], output)

    def callback(index, output, target):
        return index, 2 * output, target

    gen = (run.results for run in runs.values())
    results = concatenate(gen, reduction="mean", callback=callback)
    assert np.allclose(2 * outputs[0], results.test.output)
