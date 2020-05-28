from tensorflow import keras
from tensorflow.keras.layers import Dense


def create_model(hidden_sizes):
    layers = [Dense(hidden_sizes[0], activation="relu", input_shape=[2])]
    for hidden_size in hidden_sizes[1:]:
        layers.append(Dense(hidden_size, activation="relu"))
    layers.append(Dense(1))
    return keras.Sequential(layers)
