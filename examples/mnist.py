from dataclasses import dataclass

import numpy as np
from tensorflow import keras

import ivory
import ivory.core.data
from ivory.utils.fold import stratified_kfold_split

fashion_mnist = keras.datasets.fashion_mnist

(TRAIN_IMAGES, TRAIN_LABELS), (TEST_IMAGES, TEST_LABELS) = fashion_mnist.load_data()
TRAIN_IMAGES = TRAIN_IMAGES / 255.0
TEST_IMAGES = TEST_IMAGES / 255.0


@dataclass(repr=False)
class Data(ivory.core.data.Data):
    def __post_init__(self):
        super().__post_init__()
        self.input = np.vstack((TRAIN_IMAGES, TEST_IMAGES))
        self.target = np.concatenate((TRAIN_LABELS, TEST_LABELS))
        self.index = np.arange(len(self.input))
        fold = stratified_kfold_split(TRAIN_LABELS, n_splits=5)
        self.fold = np.concatenate([fold, np.full(TEST_IMAGES.shape[0], -1)])


def create_model():
    layers = [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
    model = keras.Sequential(layers)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
