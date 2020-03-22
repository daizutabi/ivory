import os
import pickle
import tempfile

from ivory import utils
from ivory.core.base import CallbackCaller
from ivory.core.instance import create_base_instance


class Run(CallbackCaller):
    __slots__ = []  # type:ignore

    def set_tracking(self, tracker, experiment_id, param_names=None):
        if not self.id:
            self.id = tracker.create_run(experiment_id, self.name, self.source_name)
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
        for key, state_dict in self.state_dict().items():
            path = os.path.join(directory, f"{key}.pickle")
            with open(path, "wb") as file:
                pickle.dump(state_dict, file)

    def load(self, directory):
        state_dict = {}
        for path in os.listdir(directory):
            if path.endswith(".pickle"):
                name = path.split(".")[0]
                with open(path, "rb") as file:
                    state_dict[name] = pickle.load(file)
        return state_dict


def create_run(params, source_name=""):
    return create_base_instance(params, "run", source_name=source_name)


def start(params):
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
        run = create_run(params)
    state_dict = run.load(epoch_path)
    run.load_state_dict(state_dict)
    return run
