from dataclasses import dataclass

import tensorflow as tf

import ivory.core.data


@dataclass(repr=False)
class Dataset(ivory.core.data.Dataset):
    batch_size: int = 32
    shuffle: bool = False

    @property
    def dataset(self):
        index, input, target = self[:]
        dataset = tf.data.Dataset.from_tensor_slices((input, target))
        dataset = dataset.batch(self.batch_size)
        if self.shuffle:
            dataset = dataset.shuffle(len(index))
        return dataset


@dataclass(repr=False)
class Datasets(ivory.core.data.Datasets):
    batch_size: int = 32

    def get_dataset(self, mode):
        shuffle = True if mode == "train" else False
        return self.dataset(
            self.data, mode, self.fold, batch_size=self.batch_size, shuffle=shuffle
        )
