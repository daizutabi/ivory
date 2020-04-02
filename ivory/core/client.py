import ivory.core.ui
from ivory import utils
from ivory.core.base import Base
from ivory.core.experiment import Experiment
from ivory.core.instance import create_base_instance


class Client(Base):
    def create_experiment(self, path: str) -> Experiment:
        params, source_name = utils.load_params(path, self.source_name)
        experiment = create_base_instance(params, "experiment", source_name)
        experiment.set_client(self)
        return experiment

    def ui(self):
        ivory.core.ui.run(self.tracker.tracking_uri)


def create_client(path="client", directory=".") -> Client:
    source_name = utils.normpath(path, directory)
    params, _ = utils.load_params(source_name)
    with utils.chdir(source_name):
        client = create_base_instance(params, "client", source_name)
    return client
