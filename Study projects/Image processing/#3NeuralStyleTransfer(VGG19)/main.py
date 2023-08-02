import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import layers

from matplotlib import pyplot as plt
from matplotlib import image


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


def deprocess_image(image_tensor):
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


def load_image_to_vgg19_with_validation(loaded_image_description):
    while True:
        path_to_image = input(loaded_image_description)

        loaded_image, error_id = load_image_to_vgg19(path_to_image)

        if error_id == 0:
            break
        elif error_id == 1:
            print("Image have no correct size. Load image with size 224x224.")
        elif error_id == 2:
            print("Wrong image path.")

    return loaded_image, path_to_image


def show_image_tensor(image_tensor):
    image_tensor = tf.squeeze(image_tensor, axis=0)

    plt.imshow(image_tensor)
    plt.axis("off")
    plt.show(block=True)

    return image_tensor


def get_style_vgg19(vgg: keras.applications.VGG19):
    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]

    model = keras.Model(inputs=vgg.input, outputs=[content_outputs, style_outputs])
    return model


def to_standart(tensor):
    return (tensor - tf.reduce_mean(tensor)) / tf.math.reduce_std(tensor)


def to_normalize(tensor):
    min_value = tf.reduce_min(tensor)
    max_value = tf.reduce_max(tensor)
    return (tensor - min_value) / (max_value - min_value)


def get_gram_matrix(features):
    a = tf.reshape(features, (-1, features.shape[-1]))
    return to_standart(tf.matmul(a, a, transpose_a=True) / float(a.shape[0]))


def get_content_loss(image_content_outputs, source_image_content_outputs):
    content_loss = 0
    for i in range(len(image_content_outputs)):
        content_loss += tf.reduce_mean(to_normalize((image_content_outputs[i] - source_image_content_outputs[i]) ** 2))
    content_loss /= len(image_content_outputs)
    return content_loss


def get_style_loss(image_style_outputs, style_image_style_outputs, **kwargs):
    gram_params = [1.0 / len(image_style_outputs)] * len(image_style_outputs)

    if 'gram_params' in kwargs:
        gram_params = kwargs['gram_params']

    style_loss = 0
    for i in range(len(image_style_outputs)):
        style_loss += (tf.reduce_mean((get_gram_matrix(image_style_outputs[i]) - get_gram_matrix(style_image_style_outputs[i])) ** 2)) * gram_params[i]

    return style_loss


def apply_style_to_image(image_tensor, style_image_tensor, lr, content_param, style_param, style_vgg: keras.Model, **kwargs):
    image_tensor = keras.applications.vgg19.preprocess_input(image_tensor)
    start_tensor = image_tensor
    if 'start_image' in kwargs:
        start_tensor = keras.applications.vgg19.preprocess_input(kwargs['start_image'])
    style_image_tensor = keras.applications.vgg19.preprocess_input(style_image_tensor)

    m, d = get_parameters_vgg_processing()
    m_tensor = tf.ones_like(start_tensor) * [m[2], m[1], m[0]]

    weights = tf.Variable(-tf.math.log(255 / (start_tensor + m_tensor) - 1), trainable=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    count_bad_epochs_to_break = 5
    min_difference_to_break = lr * 0.1
    content_norm_param = 0

    last_losses = []
    epoch_id = 0
    min_loss = 999
    best_image = None
    while True:
        with tf.GradientTape() as tape:
            tape.watch(weights)
            input_image = -m_tensor + tf.sigmoid(weights) * 255

            source_image_content_outputs, _ = style_vgg(image_tensor, training=True)
            image_content_outputs, image_style_outputs = style_vgg(input_image, training=True)
            _, style_image_style_outputs = style_vgg(style_image_tensor, training=True)

            content_loss = get_content_loss(image_content_outputs, source_image_content_outputs)
            style_loss = get_style_loss(image_style_outputs, style_image_style_outputs)

            if epoch_id == 0:
                content_norm_param = 1 / content_loss

            loss = content_loss * content_norm_param * content_param + style_loss * style_param
#show_image_tensor(tf.cast(deprocess_image(input_image), dtype=tf.uint8))
            if loss < min_loss:
                min_loss = loss
                best_image = input_image

        last_losses.append(loss)
        if epoch_id >= count_bad_epochs_to_break:
            last_losses.pop(0)

        print(f"{epoch_id}: {loss}")
        gradients = tape.gradient(loss, weights)
        optimizer.apply_gradients([(gradients, weights)])

        # if epoch_id >= count_bad_epochs_to_break - 1 and abs(
        #         last_losses[-1] - last_losses[0]) < min_difference_to_break:
        #     if loss < 0.2:
        #         break

        a = False
        if a:
            break

        epoch_id += 1

    output_image = tf.Variable(best_image)
    image_tensor = deprocess_image(output_image)
    image_tensor = tf.where(image_tensor > 255, tf.ones_like(image_tensor) * 255, image_tensor)
    image_tensor = tf.where(image_tensor < 0, tf.zeros_like(image_tensor), image_tensor)
    image_tensor = tf.cast(image_tensor, dtype=tf.uint8)
    return image_tensor


vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet")
style_vgg19 = get_style_vgg19(vgg19)


previous_path_to_save_image = "some_image.png"
path_to_save_with_id = "some_image.png"
saved_image_id = 0
while True:
    loaded_image, path_to_image = load_image_to_vgg19_with_validation("Enter a path to source image: ")
    loaded_style_image, _ = load_image_to_vgg19_with_validation("Enter a path to style image: ")
    to_load_start_image = input("Do you need to load start image, that will be change to source image with style? (yes/no)") == "yes"

    start_image = loaded_image
    if to_load_start_image:
        start_image, _ = load_image_to_vgg19_with_validation("Enter a path to start image: ")
    print()

    content_param = float(input("Enter the content param: "))
    style_param = float(input("Enter the style param: "))

    processed_image = apply_style_to_image(loaded_image, loaded_style_image, 0.1, content_param, style_param, style_vgg19, start_image=start_image)
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
