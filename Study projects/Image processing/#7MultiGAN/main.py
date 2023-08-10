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

import math


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

count_previous_generated_images = 2
count_epochs_between_previous_generated_images = 100
previous_generated_images = []

min_gen_loss_value = 0.6
max_gen_loss_value = 1.0

count_discriminator_first_epochs = 1

start_gen0_loss = 100
start_gen1_loss = 100

start_gen0_diff = 0
start_gen1_diff = 0

diff_gen_0_list = list()
diff_disc_0_list = list()


def fit_generator_and_discriminator(generator, discriminator, optimizers, train_x, train_y, epochs, verbose=1, global_num_epoch=1):
    history = []
    global f_save_images
    global count_generator_epochs
    global min_gen_loss_value
    global max_gen_loss_value
    global start_gen0_loss
    global start_gen1_loss
    global start_gen0_diff
    global start_gen1_diff
    global diff_gen_0_list
    global diff_disc_0_list

    generator_optimizers, discriminator_optimizers = optimizers

    if global_num_epoch == 0:
        count_generator_epochs = 0

    for e in range(epochs):
        with tf.GradientTape() as gen_tape0, tf.GradientTape() as gen_tape1, tf.GradientTape() as disc_tape0, tf.GradientTape() as disc_tape1:
            generated_images0 = generator[0](train_x, training=True)
            generated_images1 = generator[1](train_x, training=True)
            generated_images = [generated_images0, generated_images1]

            if global_num_epoch % count_epochs_between_previous_generated_images == 0:
                previous_generated_images.append(generated_images[random.randint(0, 1)])
                if len(previous_generated_images) > count_previous_generated_images:
                    previous_generated_images.pop(0)

            #################################################################################################
            d_fake0_list = list()
            d_fake0_list.append(discriminator[0]([generated_images0, train_x], training=True))
            d_fake0_list.append(discriminator[0]([generated_images1, train_x], training=True))
            for previous_images in previous_generated_images:
                d_fake0_list.append(discriminator[0]([previous_images, train_x], training=True))

            d_fake0_unstack = list()
            for fake_images in d_fake0_list:
                d_fake0_unstack.append(tf.unstack(fake_images, axis=0))

            d_fake0 = tf.stack(d_fake0_unstack, axis=0)
            #################################################################################################
            d_fake1_list = list()
            d_fake1_list.append(discriminator[1]([generated_images0, train_x], training=True))
            d_fake1_list.append(discriminator[1]([generated_images1, train_x], training=True))
            for previous_images in previous_generated_images:
                d_fake1_list.append(discriminator[1]([previous_images, train_x], training=True))

            d_fake1_unstack = list()
            for fake_images in d_fake1_list:
                d_fake1_unstack.append(tf.unstack(fake_images, axis=0))

            d_fake1 = tf.stack(d_fake1_unstack, axis=0)
            #################################################################################################

            d_real0 = discriminator[0]([train_y, train_x], training=True)
            d_real1 = discriminator[1]([train_y, train_x], training=True)

            disc_real_loss0 = tf.reduce_mean(cross_entropy(tf.zeros_like(d_real0), d_real0))
            disc_fake_loss0 = tf.reduce_mean(cross_entropy(tf.ones_like(d_fake0), d_fake0))
            disc_loss0 = (disc_real_loss0 + disc_fake_loss0) / 2

            disc_real_loss1 = tf.reduce_mean(cross_entropy(tf.zeros_like(d_real1), d_real1))
            disc_fake_loss1 = tf.reduce_mean(cross_entropy(tf.ones_like(d_fake1), d_fake1))
            disc_loss1 = (disc_real_loss1 + disc_fake_loss1) / 2

            gen_loss0 = tf.reduce_mean(cross_entropy(tf.zeros_like(d_fake0_list[0]), d_fake0_list[0]))
            gen_loss1 = tf.reduce_mean(cross_entropy(tf.zeros_like(d_fake1_list[1]), d_fake1_list[1]))

            gen_loss1 *= 0
            disc_loss1 *= 0

            if global_num_epoch == count_discriminator_first_epochs:
                start_gen0_loss = float(gen_loss0)
                start_gen1_loss = float(gen_loss1)

            if global_num_epoch < count_discriminator_first_epochs:
                gen_loss0 *= 0
                gen_loss1 *= 0

            gan_loss = tf.reduce_mean(tf.losses.mean_squared_error(train_y, generated_images0))

        #generator_gan_gradient = gen_tape.gradient(gan_loss, generator.trainable_variables)

        # show_image_tensor(tf.expand_dims(generated_images[0], axis=0))

        mean_disc_loss = (disc_loss0 + disc_loss1) / 2
        mean_gen_loss = (gen_loss0 + gen_loss1) / 2

        disc_loss0_local = (disc_real_loss0 + tf.reduce_mean(cross_entropy(tf.ones_like(d_fake0_list[0]), d_fake0_list[0]))) / 2
        disc_loss1_local = (disc_real_loss1 + tf.reduce_mean(cross_entropy(tf.ones_like(d_fake1_list[1]), d_fake1_list[1]))) / 2

        generator_gradient0 = gen_tape0.gradient(gen_loss0, generator[0].trainable_variables)
        discriminator_gradient0 = disc_tape0.gradient(disc_loss0, discriminator[0].trainable_variables)

        generator_gradient1 = gen_tape1.gradient(gen_loss1, generator[1].trainable_variables)
        discriminator_gradient1 = disc_tape1.gradient(disc_loss1, discriminator[1].trainable_variables)

        diff_gen_0 = 0
        diff_gen_1 = 0
        diff_disc_0 = 0
        diff_disc_1 = 0

        for gradient_id in range(len(generator_gradient0)):
            diff_gen_0 += tf.matmul(tf.reshape(generator_gradient0[gradient_id], (1, -1)), tf.reshape(generator_gradient0[gradient_id], (-1, 1)))
            diff_gen_1 += tf.matmul(tf.reshape(generator_gradient1[gradient_id], (1, -1)), tf.reshape(generator_gradient1[gradient_id], (-1, 1)))

        for gradient_id in range(len(discriminator_gradient0)):
            diff_disc_0 += tf.matmul(tf.reshape(discriminator_gradient0[gradient_id], (1, -1)), tf.reshape(discriminator_gradient0[gradient_id], (-1, 1)))
            diff_disc_1 += tf.matmul(tf.reshape(discriminator_gradient1[gradient_id], (1, -1)), tf.reshape(discriminator_gradient1[gradient_id], (-1, 1)))

        diff_gen_0 = float(diff_gen_0)
        diff_gen_1 = float(diff_gen_1)

        diff_disc_0 = float(diff_disc_0)
        diff_disc_1 = float(diff_disc_1)

        diff_gen_0_list.append(diff_gen_0)
        diff_disc_0_list.append(diff_disc_0)

        if global_num_epoch == count_discriminator_first_epochs:
            start_gen0_diff = float(diff_gen_0)
            start_gen1_diff = float(diff_gen_1)

        max_diff = 0.05

        if diff_gen_0 >= max_diff or diff_disc_0 >= max_diff:
            pass

        if not (diff_gen_0 < max_diff and diff_disc_0 < max_diff):
            pass

# or (diff_gen_0 >= max_diff or diff_disc_0 >= max_diff)
        if gen_loss0 >= start_gen0_loss or (diff_gen_0 >= max_diff or diff_disc_0 >= max_diff):
            generator_optimizers[0].apply_gradients(zip(generator_gradient0, generator[0].trainable_variables))

            if global_num_epoch % f_save_images == 0:
                show_image_tensor(tf.expand_dims(generated_images0[0], axis=0), path_to_save_image=f"results\image_epoch_{global_num_epoch}.png", to_show_image=False)

            count_generator_epochs += 1
# and (diff_gen_0 < max_diff and diff_disc_0 < max_diff)
        if gen_loss0 < start_gen0_loss and (diff_gen_0 < max_diff and diff_disc_0 < max_diff) or global_num_epoch < count_discriminator_first_epochs:
            discriminator_optimizers[0].apply_gradients(zip(discriminator_gradient0, discriminator[0].trainable_variables))
# or (diff_gen_1 >= max_diff or diff_disc_1 >= max_diff)
        if gen_loss1 >= start_gen1_loss:
            generator_optimizers[1].apply_gradients(zip(generator_gradient1, generator[1].trainable_variables))
# and (diff_gen_1 < max_diff and diff_disc_1 < max_diff)
        if gen_loss1 < start_gen1_loss  or global_num_epoch < count_discriminator_first_epochs:
            discriminator_optimizers[1].apply_gradients(zip(discriminator_gradient1, discriminator[1].trainable_variables))

        #generator_optimizer.apply_gradients(zip(generator_gan_gradient, generator.trainable_variables))

        if verbose == 1:
            print(f"Epoch {epochs}: gen_loss = {mean_gen_loss}, disc_loss = {mean_disc_loss}")

        history.append([gan_loss, gen_loss0, disc_loss0])
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


count_input_neurons = 8

images = load_image_dataset("dataset", 16, 16)
images = tf.stack(tf.unstack(images, axis=0)[:30])
random_input = tf.random.uniform((images.shape[0], count_input_neurons), minval=0, maxval=1)

to_load_model = input("Do you need load model from file? (y/n)") == "y"
if to_load_model:
    path_to_model = input("Enter the path to the model: ")
    generator, discriminator = load_generator_and_discriminator(path_to_model)
else:
    generator0, discriminator0 = get_generator_and_discriminator(count_input_neurons)
    generator1, discriminator1 = get_generator_and_discriminator(count_input_neurons)

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

    generator_optimizer0 = keras.optimizers.Adam(learning_rate=0.00004)
    generator_optimizer1 = keras.optimizers.Adam(learning_rate=0.00004)

    discriminator_optimizer0 = keras.optimizers.Adam(learning_rate=0.00004)
    discriminator_optimizer1 = keras.optimizers.Adam(learning_rate=0.00004)

    optimizers = ((generator_optimizer0, generator_optimizer1), (discriminator_optimizer0, discriminator_optimizer1))

    #fit_generator_to_produce_no_equal_images(generator, generator_compare_optimizer, random_input, 5, 5000, verbose = 1)

    to_train_generator = False
    epoch = 0
    while True:
        history = fit_generator_and_discriminator([generator0, generator1], [discriminator0, discriminator1], optimizers, random_input, images, 1, verbose=0, global_num_epoch=epoch)

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
                save_generator_and_discriminator(generator0, discriminator0, path_to_save_model)

        epoch += 1

    to_save_model = input("Do you want to save model? (y/n)") == "y"
    if to_save_model:
        path_to_save_model = input("Enter the path to save the model: ")
        save_generator_and_discriminator(generator0, discriminator0, path_to_save_model)

print()

while True:
    to_take_sample = input("Do you want take sample data? (y/n)") == "y"
    if to_take_sample:
        sample_id = int(input(f"Sample id (count samples is {images.shape[0]}): "))
        prediction = generator0.predict(tf.expand_dims(random_input[sample_id], axis=0))
        show_image_tensor(prediction)
    else:
        input_tensor = tf.random.uniform((1, count_input_neurons), minval=0, maxval=1)
        prediction = generator0.predict(input_tensor)
        show_image_tensor(prediction)
    print()