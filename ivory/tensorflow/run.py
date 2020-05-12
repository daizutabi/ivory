import os

import tensorflow as tf
from tensorflow import keras

import ivory.core.run


class Run(ivory.core.run.Run):
    def save(self, directory: str):
        super().save(directory)
        if self.model:
            path = os.path.join(directory, "model")
            level = tf.get_logger().level
            tf.get_logger().setLevel("WARNING")
            self.model.save(path)
            tf.get_logger().setLevel(level)

    def load_instance(self, path):
        self.model = keras.models.load_model(path)
