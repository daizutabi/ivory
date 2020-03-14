# import os
#
#
# def test_tracking(run):
#     class Experiment:
#         def __init__(self):
#             self.name = "abc"
#
#     experiment = Experiment()
#     tracking = run.tracking
#     tracking.on_experiment_start(experiment)
#     tracking.on_fit_start(run)
#     assert isinstance(tracking.run_id, str)
#     assert os.path.exists(os.path.join(tracking.directory, "params.yaml"))
#     assert os.path.exists(os.path.join(tracking.directory, "current"))
#     os.remove(os.path.join(tracking.directory, "params.yaml"))  # this yaml is invalid.
#
#     run.metrics.record = {"x": 1}
#     run.metrics.epoch = 1
#     run.monitor.score = 1
#     run.monitor.best_score = 1
#     run.monitor.best_epoch = 1
#     run.monitor.is_best = True
#     tracking.on_epoch_end(run)
#     assert os.path.exists(os.path.join(tracking.directory, "best"))
#     tracking.on_epoch_end(run)
#     print(os.listdir(tracking.directory))
#     # tracking.on_fit_end(run)
