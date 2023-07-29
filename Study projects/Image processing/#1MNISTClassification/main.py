import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    return (x_train, y_train_cat), (x_test, y_test_cat)


def create_model(size_image_x, size_image_y, kernel_size, max_pool_kernel_size, filters_conv_1, filters_conv_2, n_dense_1):
    input_image = layers.Input(shape=(size_image_x, size_image_y, 1))

    conv_1 = layers.Conv2D(filters_conv_1, (kernel_size, kernel_size), padding="same", activation="relu")(input_image)
    max_pooling_1 = layers.MaxPool2D((max_pool_kernel_size, max_pool_kernel_size), strides=max_pool_kernel_size)(conv_1)

    conv_2 = layers.Conv2D(filters_conv_2, (kernel_size, kernel_size), padding="same", activation="relu")(max_pooling_1)

    flatten = layers.Flatten()(conv_2)

    dense_1 = layers.Dense(units=n_dense_1, activation="relu")(flatten)
    dense_2 = layers.Dense(units=10, activation="softmax")(dense_1)

    model = keras.Model(inputs=input_image, outputs=dense_2)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


size_image_x = 28
size_image_y = 28
kernel_size = 3
max_pool_kernel_size = 2
filters_conv_1 = 64
filters_conv_2 = 32
n_dense_1 = 128

(x_train, y_train_cat), (x_test, y_test_cat) = get_data()
model = create_model(size_image_x, size_image_y, kernel_size, max_pool_kernel_size, filters_conv_1, filters_conv_2, n_dense_1)

print("Training the model: ")
model.fit(x_train, y_train_cat, batch_size=32, epochs=5)

print("\n========================================\n")
print("Evaluating the model:")

model.evaluate(x_test, y_test_cat)
