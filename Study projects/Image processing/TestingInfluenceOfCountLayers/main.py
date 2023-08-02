import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np


class MyStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=1e-5):
        super(MyStoppingCallback, self).__init__()
        self.threshold = threshold
        self.prev_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        # Получаем текущее значение функции потерь на валидационном наборе данных
        loss = logs['loss']

        # Проверка критерия остановки
        if self.prev_loss - loss < self.threshold:
            self.model.stop_training = True

        self.prev_loss = loss


def get_data(min_x, max_x, min_y, max_y, count_samples):
    x_data = tf.reshape(tf.linspace(min_x, max_x, count_samples), (count_samples, 1))
    y_data = (tf.sin(x_data) + 1) / 2
    return x_data, y_data


def get_model(count_hidden_neurons, count_hidden_layers):
    input_layer = layers.Input(shape=1)
    x = input_layer

    for i in range(count_hidden_layers):
        x = layers.Dense(units=count_hidden_layers, activation="sigmoid")(x)

    output_layer = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse")

    return model


x_data, y_data = get_data(min_x=-5, max_x=5, min_y=0, max_y=1, count_samples=5000)

loses_with_layers = []
end_count_hidden_layers = 15
for count_hidden_layers in range(1, end_count_hidden_layers):
    model = get_model(count_hidden_neurons=1 + (count_hidden_layers - 1) * 10, count_hidden_layers=1)
    start_loss = model.evaluate(x_data, y_data)
    history = model.fit(x_data, y_data, batch_size=32, epochs=30, verbose=0)
    print(f"Loss with {count_hidden_layers} hidden layers: {history.history['loss'][-1]}")
