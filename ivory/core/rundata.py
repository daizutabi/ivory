from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

import ivory.core.data
from ivory import utils
from ivory.core.experiment import Experiment
from ivory.core.instance import create_base_instance


@dataclass
class RunData(ivory.core.data.Data):
    mode: str = "eval"
    path: str = ""

    def init(self, run):
        self.experiments: List[Experiment] = []
        it = utils.params_iter(run.source_name, self.path)
        for params, source_name in it:
            if source_name == run.source_name:
                continue
            tracker = run.tracking.create_tracker()
            if tracker.get_experiment_id(params["experiment"]["name"]):
                experiments = create_base_instance(params, "experiment", source_name)
                experiments.set_tracker(tracker)
                self.experiments.append(experiments)

    def get(self, **query):
        val = dict(output=defaultdict(list), target=defaultdict(list))
        test = dict(output=defaultdict(list))
        data = dict(val=val, test=test)
        for experiment in self.experiments:
            for run_id in experiment.search_run(**query):
                results = experiment.load_instance(run_id, "results")
                for mode in ["val", "test"]:
                    d = results[mode]
                    self.reshape(d)
                    for index, output, *target in zip(*d.values()):
                        data[mode]['output'][index].append(output)
                        if target:
                            data[mode]['target'][index].append(*target)
        self.data = data

    def reshape(self, data):
        data["index"] = data["index"].reshape(-1)
        for name in ["output", "target"]:
            if name in data and data[name].ndim == 3:
                value = data[name].transpose(0, 2, 1)
                data[name] = value.reshape((-1, value.shape[-1]))


# @dataclass
# class Data:
#     mode: str = "train"
#     initialized: bool = False
#     fold: Any = None
#
#     def initialize(self):
#         self.init()
#         self.initialized = True
#
#     def init(self):
#         """Initializes the data. For example, read a csv file as a DataFrame.
#
#         Called from ivory.core.data.Data.
#         """
#         raise NotImplementedError
#
#     def get(self, index=None):
#         """Returns a subset of data according to `mode` and `index`.
#
#         Returned object can be any type but should be processed by Dataset's ``get()``.
#
#         Args:
#             index (list): 1d-array of bool, optional. The length is the same as `fold`.
#
#         Called from ivory.core.data.DataLoaders.
#         """
#         raise NotImplementedError
#
#
#
# @dataclass
# class Dataset:
#     mode: str
#     data: Any
#     transform: Optional[Callable] = None
#
#     def __repr__(self):
#         cls_name = self.__class__.__name__
#         return f"{cls_name}(mode={self.mode}, num_samples={len(self)})"
#
#     def __len__(self):
#         raise NotImplementedError
#
#     def __getitem__(self, index):
#         if index >= len(self):
#             raise IndexError
#         index, input, *target = self.get(index)
#         if self.transform:
#             input, *target = self.transform(self.mode, input, *target)
#         return [index, input, *target]
#
#     def get(self, index):
#         """Returns a tuple of (index, input, target) or (index, input)."""
#         raise NotImplementedError
#
#
# @dataclass
# class DataLoaders(Dict):
#     dataset: Callable
#     fold: int = 0
#     batch_size: int = 32
#
#     def __repr__(self):
#         cls_name = self.__class__.__name__
#         if isinstance(self.dataset, functools.partial):
#             dataset = self.dataset.func.__module__
#             dataset += "." + self.dataset.func.__name__
#             kwargs = [f"{key}={value}" for key, value in self.dataset.keywords.items()]
#             kwargs = ", ".join(kwargs)
#         else:
#             dataset = self.dataset.__module__
#             dataset += "." + self.dataset.__name__
#             kwargs = ""
#         s = f"{cls_name}(dataset={dataset}({kwargs}), fold={self.fold}, "
#         return s + f"batch_size={self.batch_size})"
#
#     def init(self, data: Data, run=None):
#         if not data.initialized:
#             data.initialize(run)
#         if data.mode == "train":
#             for mode in ["train", "val"]:
#                 index = self.get_index(mode, data)
#                 dataset = self.dataset(mode, data.get(index))
#                 self[mode] = self.get_dataloader(mode, dataset)
#         elif data.mode == "test":
#             index = self.get_index("test", data)
#             dataset = self.dataset("test", data.get(index))
#             self["test"] = self.get_dataloader("test", dataset)
#         else:
#             raise ValueError(f"Unknown mode: {data.mode}")
#
#     def get_index(self, mode, data):
#         if mode == "train":
#             return data.fold != self.fold
#         elif mode == "val":
#             return data.fold == self.fold
#         elif mode == "test":
#             return
#
#     def get_dataloader(self, mode, dataset):
#         return dataset
