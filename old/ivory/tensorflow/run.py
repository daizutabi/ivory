import os

import tensorflow as tf

import ivory.core.run


class Run(ivory.core.run.Run):
    def save(self, directory: str):
        super().save(directory)
        if self.model:
            path = os.path.join(directory, "model/checkpoint")
            level = tf.get_logger().level
            tf.get_logger().setLevel("WARNING")
            self.model.save_weights(path)
            tf.get_logger().setLevel(level)

    def load_instance(self, path):
        path = os.path.join(path, "checkpoint")
        if self.model:
            self.model.load_weights(path)
