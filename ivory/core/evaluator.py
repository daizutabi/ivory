from typing import Iterable, List, Tuple

from pandas import DataFrame

import ivory.utils.data
from ivory.utils.tqdm import tqdm


class Evaluator:
    def __init__(self, client):
        self.client = client
        self._run_ids: List[str] = []
        self._runs = []
        self.output = None
        self.target = None

    @property
    def run_ids(self):
        return self._run_ids

    @run_ids.setter
    def run_ids(self, run_ids: Iterable[str]):
        self._run_ids = list(run_ids)

    def load_results(self, verbose: bool = True) -> Tuple[DataFrame, DataFrame]:
        if verbose:
            run_ids = tqdm(self.run_ids, leave=False)
        client = self.client
        it = (client.load_instance(run_id, "results", "test") for run_id in run_ids)
        return ivory.utils.data.concat_results(it)

    def from_results(self, softmax=False, argmax=True, verbose: bool = True):
        output, target = self.load_results(verbose)
        if softmax:
            output = ivory.utils.data.softmax(output)
        output = ivory.utils.data.mean(output)
        target = ivory.utils.data.mean(target)
        if argmax:
            output = ivory.utils.data.argmax(output)
        self.output, self.target = output, target
        return output, target


#     def sef_runs
#
#
# def create_rfc_prob():
#     client = ivory.create_client()
#     run_ids = list(client.search_nested_run_ids("rfc"))
#     output, target = client.load_results(run_ids)
#     df = load_data("feature")
#     df["pred"] = ivory.utils.data.mean_argmax(output)
#     train = df.query("state >= 0")
#     score = f1_score(train.state, train.pred, average="macro")
#     prob = ivory.utils.data.mean(output)
#     prob.columns = [f"rfc_prob_{k}" for k in prob]
#     prob.to_feather("../input/liverpool-ion-switching/rfc_prob.feather")
#     return score
#
#
# def main():
#     create_rfc_prob()
#
#
# if __name__ == "__main__":
#     main()
