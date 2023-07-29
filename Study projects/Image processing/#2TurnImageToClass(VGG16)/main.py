import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib as plt
from matplotlib import image
from matplotlib import pyplot as plt


def get_parameters_vgg_processing():
    black_image = tf.ones((1, 1, 1, 3)) * 0
    white_image = tf.ones((1, 1, 1, 3)) * 255

    black_image_preprocessed = keras.applications.vgg19.preprocess_input(black_image)
    white_image_preprocessed = keras.applications.vgg19.preprocess_input(white_image)

    c0b = black_image_preprocessed[0][0][0][0]
    c0g = black_image_preprocessed[0][0][0][1]
    c0r = black_image_preprocessed[0][0][0][2]

    c1b = white_image_preprocessed[0][0][0][0]
    c1g = white_image_preprocessed[0][0][0][1]
    c1r = white_image_preprocessed[0][0][0][2]

    mr = c0r * (255 / (c0r - c1r))
    mg = c0g * (255 / (c0g - c1g))
    mb = c0b * (255 / (c0b - c1b))

    dr = 255 / (c1r - c0r)
    dg = 255 / (c1g - c0g)
    db = 255 / (c1b - c0b)

    return [mr, mg, mb], [dr, dg, db]


def restore_image(image_tensor):
    image_tensor_copy = tf.Variable(tf.identity(image_tensor))

    black_image = tf.ones((1, 1, 1, 3)) * 0
    white_image = tf.ones((1, 1, 1, 3)) * 255

    black_image_preprocessed = keras.applications.vgg19.preprocess_input(black_image)
    white_image_preprocessed = keras.applications.vgg19.preprocess_input(white_image)

    c0b = black_image_preprocessed[0][0][0][0]
    c0g = black_image_preprocessed[0][0][0][1]
    c0r = black_image_preprocessed[0][0][0][2]

    c1b = white_image_preprocessed[0][0][0][0]
    c1g = white_image_preprocessed[0][0][0][1]
    c1r = white_image_preprocessed[0][0][0][2]

    mr = c0r * (255 / (c0r - c1r))
    mg = c0g * (255 / (c0g - c1g))
    mb = c0b * (255 / (c0b - c1b))

    dr = 255 / (c1r - c0r)
    dg = 255 / (c1g - c0g)
    db = 255 / (c1b - c0b)

    width = image_tensor_copy.shape[1]
    height = image_tensor_copy.shape[2]

    values_to_swap = tf.identity(image_tensor_copy[:, :, :, 0])  # Копируем значения из первого канала

    # Присваиваем значения для свапа местами между каналами
    image_tensor_copy[:, :, :, 0].assign(image_tensor_copy[:, :, :, 2])
    image_tensor_copy[:, :, :, 2].assign(values_to_swap)

    mr_t = tf.one_hot(indices=tf.ones_like(tf.ones((1, width, height, 3))[:, :, :, 0], dtype=tf.int32) * 0, depth=3) * mr
    mg_t = tf.one_hot(indices=tf.ones_like(tf.ones((1, width, height, 3))[:, :, :, 0], dtype=tf.int32) * 1, depth=3) * mg
    mb_t = tf.one_hot(indices=tf.ones_like(tf.ones((1, width, height, 3))[:, :, :, 0], dtype=tf.int32) * 2, depth=3) * mb
    m_t = mr_t + mg_t + mb_t

    dr_t = tf.one_hot(indices=tf.ones_like(tf.ones((1, width, height, 3))[:, :, :, 0], dtype=tf.int32) * 0, depth=3) * dr
    dg_t = tf.one_hot(indices=tf.ones_like(tf.ones((1, width, height, 3))[:, :, :, 0], dtype=tf.int32) * 1, depth=3) * dg
    db_t = tf.one_hot(indices=tf.ones_like(tf.ones((1, width, height, 3))[:, :, :, 0], dtype=tf.int32) * 2, depth=3) * db
    d_t = dr_t + dg_t + db_t

    image_tensor_copy = (image_tensor_copy + m_t) / d_t

    return image_tensor_copy


def load_image_to_vgg19(path):
    try:
        loaded_image = image.imread(path)
    except FileNotFoundError:
        return None, 2 #Путь указан не верно

    if loaded_image.shape[0] != 224 or loaded_image.shape[1] != 224:
        return None, 1 #Неккоректный размер изображения

    loaded_image = np.expand_dims(loaded_image, axis=0)
    if loaded_image.shape[-1] > 3:
        loaded_image = tf.stack(tf.unstack(loaded_image, axis=-1)[:-1], axis=-1)

    return tf.Variable(loaded_image * 255, trainable=True), 0


def active_class_on_image(image_tensor, classes, to_reduce_loss, lr, vgg: keras.applications.VGG19, **kwargs):
    image_tensor = keras.applications.vgg19.preprocess_input(image_tensor)

    m, d = get_parameters_vgg_processing()
    m_tensor = tf.ones_like(image_tensor) * [m[2], m[1], m[0]]

    weights = tf.Variable(-tf.math.log(255 / (image_tensor + m_tensor) - 1), trainable=True)

    count_classes_of_vgg = 1000

    target_tensor = tf.zeros((1, count_classes_of_vgg))

    for class_id in classes:
        target_tensor += keras.utils.to_categorical(class_id, num_classes=count_classes_of_vgg)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    image_tensor_by_colors = tf.unstack(image_tensor, axis=-1)
    mean_brightness_r = tf.reduce_mean(image_tensor_by_colors[2])
    mean_brightness_g = tf.reduce_mean(image_tensor_by_colors[1])
    mean_brightness_b = tf.reduce_mean(image_tensor_by_colors[0])

    brightness_cof = 2
    if 'brightness_cof' in kwargs:
        brightness_cof = kwargs['brightness_cof']

    count_bad_epochs_to_break = 5
    min_difference_to_break = lr * 0.1

    count_bad_epochs = 0
    last_losses = []
    epoch_id = 0
    while count_bad_epochs < count_bad_epochs_to_break:
        with tf.GradientTape() as tape:
            tape.watch(weights)
            input_image = -m_tensor + tf.sigmoid(weights) * 255

            prediction = vgg(input_image, training=True)
            loss = tf.reduce_sum((prediction - target_tensor) ** 2)
            if not to_reduce_loss:
                input_image_by_colors = tf.unstack(image_tensor, axis=-1)
                input_mean_brightness_r = tf.reduce_mean(input_image_by_colors[2])
                input_mean_brightness_g = tf.reduce_mean(input_image_by_colors[1])
                input_mean_brightness_b = tf.reduce_mean(input_image_by_colors[0])
                brightness_loss  = ((input_mean_brightness_r - mean_brightness_r) ** 2)
                brightness_loss += ((input_mean_brightness_g - mean_brightness_g) ** 2)
                brightness_loss += ((input_mean_brightness_b - mean_brightness_b) ** 2)
                loss += brightness_loss * brightness_cof

        last_losses.append(loss)
        if epoch_id >= count_bad_epochs_to_break:
            last_losses.pop(0)

        print(f"{epoch_id}: {loss}")
        gradients = tape.gradient(loss, weights)
        if not to_reduce_loss:
            gradients *= -1
        optimizer.apply_gradients([(gradients, weights)])


        if epoch_id >= count_bad_epochs_to_break - 1 and abs(last_losses[-1] - last_losses[0]) < min_difference_to_break:
            if (to_reduce_loss and loss < 0.2) or (not to_reduce_loss and loss >= 1):
                break

        epoch_id += 1

    output_image = tf.Variable(-m_tensor + tf.sigmoid(weights) * 255)
    image_tensor = restore_image(output_image)
    image_tensor = tf.where(image_tensor > 255, tf.ones_like(image_tensor) * 255, image_tensor)
    image_tensor = tf.where(image_tensor < 0, tf.zeros_like(image_tensor), image_tensor)
    image_tensor = tf.cast(image_tensor, dtype=tf.uint8)
    return image_tensor


def show_image_tensor(image_tensor):
    image_tensor = tf.squeeze(image_tensor, axis=0)

    plt.imshow(image_tensor)
    plt.axis("off")
    plt.show(block=True)

    return image_tensor


vgg19 = keras.applications.VGG19(include_top=True, weights="imagenet")

previous_path_to_save_image = "some_image.png"
path_to_save_with_id = "some_image.png"
saved_image_id = 0
while True:
    while True:
        path_to_image = input("Enter a path to image: ")

        loaded_image, error_id = load_image_to_vgg19(path_to_image)

        if error_id == 0:
            break
        elif error_id == 1:
            print("Image have no correct size. Load image with size 224x224.")
        elif error_id == 2:
            print("Wrong image path.")

        print()
    print()

    #path_to_save_image = input("Enter a path to save processed image: ")

    classes = list(map(int, input("Enter classes, that you want to (see/don't see) on image: ").split(" ")))
    to_reduce_loss = True if input("Do you want to see or don't see classes on image? (yes/not)") == "yes" else False

    processed_image = active_class_on_image(loaded_image, classes, to_reduce_loss, 0.1, vgg19)
    image_to_save = show_image_tensor(processed_image)
    print()

    print("Enter the path to save the image.")
    print("1. If you don't want to save the image - enter symbol 'n'.")
    print("2. If you want to save image with the save path - enter symbol 's'")
    print("3. If you want to save image next id - enter symbol 'i'. You can write like this: \"path.png i\". In this"
          " case current image save with id.")
    path_to_save_image = input("The path to the image: ")

    if previous_path_to_save_image == "":
        previous_path_to_save_image = path_to_save_image

    if path_to_save_image != "n":
        image_pil = Image.fromarray(tf.transpose(image_to_save, perm=[0, 1, 2]).numpy())
        path = ""

        have_sub_command_i = len(path_to_save_image.split(" ")) > 1 and path_to_save_image.split(" ")[1]

        if have_sub_command_i:
            path_to_save_image = path_to_save_image.split(" ")[0]
            saved_image_id = 0
            path_to_save_with_id = path_to_save_image

        if path_to_save_image == "s":
            path = previous_path_to_save_image
        elif path_to_save_image == "i" or have_sub_command_i:
            extension_pointer = path_to_save_with_id.rfind(".")
            path = path_to_save_with_id[:extension_pointer] + f"_{saved_image_id}" + path_to_save_with_id[extension_pointer:]
            saved_image_id += 1
        else:
            path = path_to_save_image
            path_to_save_with_id = path_to_save_image
            saved_image_id = 0

        image_pil.save(path)
        previous_path_to_save_image = path
    print()
