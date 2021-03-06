from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

import ivory
import ivory.callbacks.metrics
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
        self.input = np.vstack((TRAIN_IMAGES[:1000], TEST_IMAGES[:100]))
        self.target = np.concatenate((TRAIN_LABELS[:1000], TEST_LABELS[:100]))
        self.index = np.arange(len(self.input))
        fold = stratified_kfold_split(TRAIN_LABELS[:1000], n_splits=5)
        self.fold = np.concatenate([fold, np.full(100, -1)])


def create_model():
    layers = [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
    return keras.Sequential(layers)


class Metrics(ivory.callbacks.metrics.Metrics):
    def call(self, output, target):
        pred = output.argmax(axis=1)
        true = target
        score = np.mean(pred == true)
        return {"score": score}


class CallbackMetric(Callback):
    def __init__(self, datasets, metrics):
        _, self.input, self.target = datasets.val[:]
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs):
        pred = self.model.predict(self.input).argmax(axis=1)
        score = np.mean(pred == self.target)
        self.metrics["callback_score"] = score


def lr_schedule(epoch):
    return epoch / 100
