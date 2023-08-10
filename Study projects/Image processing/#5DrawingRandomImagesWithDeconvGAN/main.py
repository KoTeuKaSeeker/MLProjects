import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

from matplotlib import pyplot as plt
from matplotlib import image

import keyboard


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


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


def get_generator_and_discriminator(count_input_neurons):
    # Generator
    ###########################################################
    generator_input_layer = layers.Input(shape=count_input_neurons)

    generator_dense_layer = layers.BatchNormalization()(layers.Dense(units=4 * 4 * 20, activation="sigmoid")(generator_input_layer))
    generator_dense_reshape_layer = layers.Reshape((4, 4, 20))(generator_dense_layer)

    generator_conv_layer_1 = layers.BatchNormalization()(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", activation="sigmoid")(generator_dense_reshape_layer)) # 8
    generator_conv_layer_2 = layers.BatchNormalization()(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", activation="sigmoid")(generator_conv_layer_1)) # 16
    generator_conv_layer_3 = layers.BatchNormalization()(layers.Conv2DTranspose(16, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(generator_conv_layer_2)) # 32
    generator_conv_layer_4 = layers.BatchNormalization()(layers.Conv2DTranspose(8, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(generator_conv_layer_3)) # 32
    generator_conv_layer_5_output = layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(generator_conv_layer_4) # 64

    # Discriminator
    ###########################################################
    image_shape = (generator_conv_layer_5_output.shape[1],
                   generator_conv_layer_5_output.shape[2],
                   generator_conv_layer_5_output.shape[3])

    discriminator_input_image = layers.Input(shape=image_shape)
    discriminator_input_labels = layers.Input(shape=count_input_neurons)

    discriminator_conv_layer_1 = layers.Conv2D(8, (5, 5), strides=(1, 1), padding="same", activation="relu")(discriminator_input_image) # 16
    discriminator_conv_layer_2 = layers.Conv2D(16, (5, 5), strides=(1, 1), padding="same", activation="relu")(discriminator_conv_layer_1) # 16
    discriminator_conv_layer_3 = layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu")(discriminator_conv_layer_2) # 8
    discriminator_conv_layer_4 = layers.Conv2D(48, (5, 5), strides=(2, 2), padding="same", activation="relu")(discriminator_conv_layer_3) # 4
    discriminator_conv_layer_5_output = layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="relu")(discriminator_conv_layer_4) # 4
    discriminator_flatten = layers.Flatten()(discriminator_conv_layer_5_output)

    discriminator_input_labels_flatten = layers.Flatten()(discriminator_input_labels)
    discriminator_dense_input = layers.Concatenate()([discriminator_flatten, discriminator_input_labels_flatten])

    discriminator_dense_1 = layers.Dense(units=256, activation="relu")(discriminator_dense_input)
    discriminator_dense_2 = layers.Dense(units=128, activation="relu")(discriminator_dense_1)
    discriminator_dense_3 = layers.Dense(units=32, activation="relu")(discriminator_dense_2)
    discriminator_dense_4_output = layers.Dense(units=1)(discriminator_dense_3)

    ###########################################################

    generator_model = keras.Model(generator_input_layer, generator_conv_layer_5_output)
    discriminator_model = keras.Model([discriminator_input_image, discriminator_input_labels], discriminator_dense_4_output)

    # # 0 - real, 1 - fake
    # d_real = discriminator_model([real_input_image, real_input_labels])
    # d_fake = discriminator_model([generator_model(real_input_labels), real_input_labels])
    # tf.math.log(d_real) + (1 - tf.math.log(d_fake))

    return generator_model, discriminator_model


def down_sample(filters, size, strides=2, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(keras.layers.Conv2D(filters, size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization())

    result.add(keras.layers.LeakyReLU())

    return result


def up_sample(filters, size, strides=2, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(keras.layers.Conv2DTranspose(filters, size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=True))

    result.add(keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(keras.layers.Dropout(0.5))

    result.add(keras.layers.ReLU())

    return result


def gan_dense(units, apply_batchnorm=True, target_shape=None, apply_leaky_relu=True):
    result = keras.Sequential()

    result.add(keras.layers.Dense(units=units))

    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization())

    if apply_leaky_relu:
        result.add(keras.layers.LeakyReLU())

    if target_shape is not None:
        result.add(keras.layers.Reshape(target_shape))

    return result


def get_generator_and_discriminator_v2():
    input_noise = keras.layers.Input(shape=128)

    generator_dense = gan_dense(2 * 2 * 128, target_shape=(2, 2, 128))(input_noise)

    generator_conv_1 = up_sample(512, 5, apply_dropout=True)(generator_dense) # (None, 4, 4, 512)
    generator_conv_2 = up_sample(256, 5, apply_dropout=True)(generator_conv_1) # (None, 8, 8, 256)
    generator_conv_3 = up_sample(128, 5)(generator_conv_2) # (None, 16, 16, 128)
    generator_conv_4 = up_sample(3, 5, strides=1)(generator_conv_3) # (None, 16, 16, 3)

    generator = keras.Model(inputs=input_noise, outputs=generator_conv_4)

    ##################################################################################################################

    input_image = keras.layers.Input(shape=generator_conv_4.shape[1:])
    input_labels = keras.layers.Input(shape=128)

    discriminator_conv_1 = down_sample(3, 5, strides=1, apply_batchnorm=False)(input_image) # (None, 16, 16, 3)
    discriminator_conv_2 = down_sample(128, 5)(discriminator_conv_1) # (None, 8, 8, 128)
    discriminator_conv_3 = down_sample(256, 5)(discriminator_conv_2) # (None, 4, 4, 256)
    discriminator_conv_4 = down_sample(512, 5)(discriminator_conv_3) # (None, 2, 2, 512)
    discriminator_conv_4 = keras.layers.Flatten()(discriminator_conv_4)
    discriminator_conv_4 = layers.concatenate([discriminator_conv_4, input_labels])

    discriminator_dense_1 = gan_dense(512, apply_batchnorm=False)(discriminator_conv_4)
    discriminator_dense_2 = gan_dense(256)(discriminator_dense_1)
    discriminator_dense_3 = gan_dense(64)(discriminator_dense_2)
    discriminator_dense_4 = gan_dense(1, apply_leaky_relu=False)(discriminator_dense_3)

    discriminator = keras.Model(inputs=[input_labels, input_image], outputs=discriminator_dense_4)

    return generator, discriminator


def fit_generator_to_produce_no_equal_images(generator: keras.Model, optimizer, train_x, count_images_to_compare, epochs, verbose = 1):
    for e in range(epochs):
        with tf.GradientTape() as tape:
            generated_images = generator(train_x, training=True)
            sim_loss = -tf.math.reduce_mean(tf.math.reduce_std(generated_images, axis=0))
        generator_gradient = tape.gradient(sim_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))

        a = False
        if a:
            break

        if verbose == 1:
            print(f"epoch {e}: sim_loss = {-sim_loss}")


f_save_images = 10
count_generator_epochs = 0
start_discriminator_learning_rate = 0.0001
temp_discriminator_learning_rate = -1
labmda = 100


def fit_generator_and_discriminator(generator: keras.Model, discriminator: keras.Model, optimizers, train_x, train_y, epochs, verbose=1, global_num_epoch=1):
    history = []
    min_loss_to_swap_training = 0.9
    global f_save_images
    global count_generator_epochs
    global temp_discriminator_learning_rate
    global labmda

    if global_num_epoch == 0:
        count_generator_epochs = 0
        temp_discriminator_learning_rate = float(optimizers[1].learning_rate)
        optimizers[1].learning_rate = start_discriminator_learning_rate
    for e in range(epochs):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(train_x, training=True)

            d_fake = discriminator([train_x, generated_images], training=True)
            d_real = discriminator([train_x, train_y], training=True)

            gen_loss = tf.reduce_mean(cross_entropy(tf.zeros_like(d_fake), d_fake))
            gen_l1_loss = tf.reduce_mean(tf.losses.mean_squared_error(train_y, generated_images))

            disc_real_loss = tf.reduce_mean(cross_entropy(tf.zeros_like(d_real), d_real))
            disc_fake_loss = tf.reduce_mean(cross_entropy(tf.ones_like(d_fake), d_fake))
            disc_loss = disc_real_loss + disc_fake_loss

            gen_loss_norm = gen_loss + gen_l1_loss * labmda
            disc_loss_norm = disc_loss

            gan_loss = tf.reduce_mean(tf.losses.mean_squared_error(train_y, generated_images))

        generator_optimizer, discriminator_optimizer = optimizers

        #generator_gan_gradient = gen_tape.gradient(gan_loss, generator.trainable_variables)

        # show_image_tensor(tf.expand_dims(generated_images[0], axis=0))

        discriminator_gradient = disc_tape.gradient(disc_loss_norm, discriminator.trainable_variables)
        generator_gradient = gen_tape.gradient(gen_loss_norm, generator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
        generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))

        if global_num_epoch % f_save_images == 0:
            show_image_tensor(tf.expand_dims(generated_images[0], axis=0), path_to_save_image=f"results\image_epoch_{global_num_epoch}.png", to_show_image=False)

        # if disc_loss_norm < min_loss_to_swap_training or (not gen_loss_norm < min_loss_to_swap_training and not disc_loss_norm < min_loss_to_swap_training):
        #     generator_gradient = gen_tape.gradient(gen_loss_norm, generator.trainable_variables)
        #     generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
        #
        #     if global_num_epoch % f_save_images == 0:
        #         show_image_tensor(tf.expand_dims(generated_images[0], axis=0), path_to_save_image=f"results\image_epoch_{global_num_epoch}.png", to_show_image=False)
        #
        #     count_generator_epochs += 1
        #
        # if gen_loss_norm < min_loss_to_swap_training:
        #     discriminator_gradient = disc_tape.gradient(disc_loss_norm, discriminator.trainable_variables)
        #     discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
        # else:
        #     if discriminator_optimizer.learning_rate != temp_discriminator_learning_rate:
        #         discriminator_optimizer.learning_rate = temp_discriminator_learning_rate

        #generator_optimizer.apply_gradients(zip(generator_gan_gradient, generator.trainable_variables))

        if verbose == 1:
            print(f"Epoch {epochs}: gen_loss = {gen_loss}, disc_loss = {disc_loss}")

        history.append([gan_loss, gen_loss, disc_loss])
    return history


def save_generator_and_discriminator(generator: keras.Model, discriminator: keras.Model, path_to_save):
    os.mkdir(path_to_save)

    path_to_save_generator = os.path.join(path_to_save, "generator")
    path_to_save_discriminator = os.path.join(path_to_save, "discriminator")

    generator.save(path_to_save_generator)
    discriminator.save(path_to_save_discriminator)


def load_generator_and_discriminator(path_to_load):
    path_to_load_generator = os.path.join(path_to_load, "generator")
    path_to_load_discriminator = os.path.join(path_to_load, "discriminator")

    generator = keras.models.load_model(path_to_load_generator)
    discriminator = keras.models.load_model(path_to_load_discriminator)

    return generator, discriminator


def show_image_tensor(image_tensor, path_to_save_image=None, to_show_image=True):
    image_tensor = tf.squeeze(image_tensor, axis=0)

    plt.imshow(image_tensor)
    plt.axis("off")
    if to_show_image:
        plt.show(block=True)

    if path_to_save_image is not None:
        plt.savefig(path_to_save_image)

    return image_tensor


count_input_neurons = 128

images = load_image_dataset("dataset", 16, 16)
images = tf.stack(tf.unstack(images, axis=0)[:150])
random_input = tf.random.uniform((images.shape[0], count_input_neurons), minval=0, maxval=1)

to_load_model = input("Do you need load model from file? (y/n)") == "y"
if to_load_model:
    path_to_model = input("Enter the path to the model: ")
    generator, discriminator = load_generator_and_discriminator(path_to_model)
else:
    generator, discriminator = get_generator_and_discriminator_v2()

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
    #0.00001
    generator_compare_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    generator_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    optimizers = (generator_optimizer, discriminator_optimizer)

    #fit_generator_to_produce_no_equal_images(generator, generator_compare_optimizer, random_input, 5, 5000, verbose = 1)

    to_train_generator = False
    epoch = 0
    while True:
        history = fit_generator_and_discriminator(generator, discriminator, optimizers, random_input, images, 1, verbose=0, global_num_epoch=epoch)

        print(f"epoch {epoch}: gan_loss = {history[0][0]}, gen_loss = {history[0][1]}, disc_loss = {history[0][2]}")

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
                save_generator_and_discriminator(generator, discriminator, path_to_save_model)

        epoch += 1

    to_save_model = input("Do you want to save model? (y/n)") == "y"
    if to_save_model:
        path_to_save_model = input("Enter the path to save the model: ")
        save_generator_and_discriminator(generator, discriminator, path_to_save_model)

print()

while True:
    to_take_sample = input("Do you want take sample data? (y/n)") == "y"
    if to_take_sample:
        sample_id = int(input(f"Sample id (count samples is {images.shape[0]}): "))
        prediction = generator.predict(tf.expand_dims(random_input[sample_id], axis=0))
        show_image_tensor(prediction)
    else:
        input_tensor = tf.random.uniform((1, count_input_neurons), minval=0, maxval=1)
        prediction = generator.predict(input_tensor)
        show_image_tensor(prediction)
    print()