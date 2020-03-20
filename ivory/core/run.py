import tempfile

import ivory
from ivory import utils
from ivory.core import instance
from ivory.core.base import CallbackCaller


class Run(CallbackCaller):
    __slots__ = ["library"]  # type:ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library = ""
        if "library" in self.objects:
            self.library = self.objects.pop("library")

    def set_tracking(self, tracker, experiment_id, param_names=None):
        if not self.id:
            self.id = tracker.create_run(self.name, experiment_id)
            self.params["id"] = self.id
        self.objects["tracking"] = tracker.create_tracking(experiment_id, param_names)

    def start(self):
        self.create_callbacks()
        self.trainer.fit(self)

    def state_dict(self):
        state_dict = {}
        for x in self:
            if hasattr(self[x], "state_dict"):
                state_dict[x] = self[x].state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for x in state_dict:
            if x in self:
                self[x].load_state_dict(state_dict[x])

    def save(self, directory):
        raise NotImplementedError

    def load(self, directory):
        raise NotImplementedError


create_run = instance.create_instance_factory("run")


def start(params="params.yaml"):
    if isinstance(params, str):
        params_ = utils.load_params(params)
    else:
        params_ = params
    if "environment" in params_:
        environment = ivory.create_environment(params)
    else:
        environment = None
    if "experiment" in params_ and environment:
        experiment = environment.create_experiment(params)
    elif "experiment" in params_:
        experiment = ivory.create_experiment(params)
    else:
        experiment = None
    if experiment:
        run = experiment.create_run(params)
    else:
        run = create_run(params)
    run.start()


def load_run(run_id, epoch="best", client=None, experiment=None):
    if client is None and experiment is not None and experiment.tracker:
        client = experiment.tracker.client
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = client.download_artifacts(run_id, "params.yaml", tmpdir)
        epoch_path = client.download_artifacts(run_id, epoch, tmpdir)
    params = utils.load_params(params_path)
    if experiment:
        run = experiment.create_run(params)
    else:
        run = ivory.create_run(params)
    state_dict = run.load(epoch_path)
    run.load_state_dict(state_dict)
    return run
