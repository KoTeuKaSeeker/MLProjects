import os
#os.environ['TF_CPP_MIN_LOG'] = "2"

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

from matplotlib import pyplot as plt
from matplotlib import image

import keyboard


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                images.append(img)
            except Exception as e:
                print(f"Failed to load image {filename}: {e}")
    return images


def preprocess_images(images, width, height):
    for image_id in range(len(images)):
        if images[image_id].mode != 'RGB':
            images[image_id] = images[image_id].convert('RGB')

        images[image_id] = images[image_id].resize((width, height))
    return images


def images_to_tensor(images):
    images_tensor = []
    for image in images:
        image_array = np.asarray(image)
        images_tensor.append(tf.convert_to_tensor(image_array, dtype=tf.float32) / 255)

    return tf.stack(images_tensor, axis=0)


def load_image_dataset(path_to_folder, width, height):
    images = load_images_from_folder(path_to_folder)
    images = preprocess_images(images, width, height)
    images = images_to_tensor(images)
    return images


def get_model(input_data_shape):
    input_layer = layers.Input(shape=input_data_shape)

    conv_layer_1 = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", activation="relu")(input_layer) # 16
    conv_layer_2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", activation="relu")(conv_layer_1) # 32
    conv_layer_3 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", activation="relu")(conv_layer_2) # 64
    conv_layer_4 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same", activation="relu")(conv_layer_3) # 128

    model = keras.Model(input_layer, conv_layer_4)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    return model


def show_image_tensor(image_tensor):
    image_tensor = tf.squeeze(image_tensor, axis=0)

    plt.imshow(image_tensor)
    plt.axis("off")
    plt.show(block=True)

    return image_tensor


input_dim = (8, 8, 8)

images = load_image_dataset("dataset", 128, 128)
images = tf.stack(tf.unstack(images, axis=0)[:166])
random_input = tf.random.uniform((images.shape[0], input_dim[0], input_dim[1], input_dim[2]), minval=0, maxval=1)

to_load_model = input("Do you need load model from file? (y/n)") == "y"
if to_load_model:
    path_to_model = input("Enter the path to the model: ")
    model = tf.keras.models.load_model(path_to_model)
else:
    model = get_model(input_dim)

to_train_model = input("Do you need to train model? (y/n)") == "y"
if to_train_model:
    ctrl_shift_c_is_pressed = False
    ctrl_shift_s_is_pressed = False

    def on_ctrl_shift_c_pressed():
        global ctrl_shift_c_is_pressed
        ctrl_shift_c_is_pressed = True

    def on_ctrl_shift_s_pressed():
        global ctrl_shift_s_is_pressed
        ctrl_shift_s_is_pressed = True

    keyboard.add_hotkey('ctrl+shift+c', on_ctrl_shift_c_pressed)
    keyboard.add_hotkey('ctrl+shift+s', on_ctrl_shift_s_pressed)

    while True:
        model.fit(random_input, images, batch_size=32, epochs=32)

        if ctrl_shift_c_is_pressed:
            ctrl_shift_c_is_pressed = False
            to_stop_training = input("Do you want to stop training? (y/n)") == "y"
            if to_stop_training:
                break

        if ctrl_shift_s_is_pressed:
            ctrl_shift_s_is_pressed = False
            to_save_model = input("Do you want to save model? (y/n)") == "y"
            if to_save_model:
                path_to_save_model = input("Enter the path to save the model: ")
                model.save(path_to_save_model)

    path_to_save_model = input("Enter the path to save the model: ")
    model.save(path_to_save_model)

print()

while True:
    to_take_sample = input("Do you want take sample data? (y/n)") == "y"
    if to_take_sample:
        sample_id = int(input(f"Sample id (count samples is {images.shape[0]}): "))
        prediction = model.predict(tf.expand_dims(random_input[sample_id], axis=0))
        show_image_tensor(prediction)
    else:
        input_tensor = tf.random.uniform((1, input_dim[0], input_dim[1], input_dim[2]), minval=0, maxval=1)
        prediction = model.predict(input_tensor)
        show_image_tensor(prediction)
    print()