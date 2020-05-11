import tensorflow.keras.callbacks


class Callback(tensorflow.keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run
