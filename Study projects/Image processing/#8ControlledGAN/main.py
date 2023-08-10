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
mean_squared_error = tf.keras.losses.mean_squared_error


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


def get_generator_and_discriminator_and_creator(count_input_neurons):
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

    discriminator_conv_layer_1 = layers.Conv2D(8, (5, 5), strides=(1, 1), padding="same", activation="relu")(discriminator_input_image) # 16
    discriminator_conv_layer_2 = layers.BatchNormalization()(layers.Conv2D(16, (5, 5), strides=(1, 1), padding="same", activation="relu")(discriminator_conv_layer_1)) # 16
    discriminator_conv_layer_3 = layers.BatchNormalization()(layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation="relu")(discriminator_conv_layer_2)) # 8
    discriminator_conv_layer_4 = layers.BatchNormalization()(layers.Conv2D(48, (5, 5), strides=(2, 2), padding="same", activation="sigmoid")(discriminator_conv_layer_3)) # 4

    hidden_view_dim = 32
    discriminator_hidden_view = layers.Conv2D(hidden_view_dim, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(discriminator_conv_layer_4) # 4
    discriminator_objects_to_add = layers.Conv2D(hidden_view_dim, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(discriminator_conv_layer_4) # 4
    discriminator_object_to_sub = layers.Conv2D(hidden_view_dim, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(discriminator_conv_layer_4) # 4

    # Creator
    ###########################################################

    creator_input_hidden_view = layers.Input(shape=discriminator_hidden_view.shape[1:])
    creator_conv_layer_1 = layers.Conv2DTranspose(48, (5, 5), strides=(2, 2), padding="same", activation="relu")(creator_input_hidden_view)  # 4
    creator_conv_layer_2 = layers.BatchNormalization()(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", activation="relu")(creator_conv_layer_1)) # 8
    creator_conv_layer_3 = layers.BatchNormalization()(layers.Conv2DTranspose(16, (5, 5), strides=(1, 1), padding="same", activation="relu")(creator_conv_layer_2))  # 16
    creator_conv_layer_4 = layers.BatchNormalization()(layers.Conv2DTranspose(8, (5, 5), strides=(1, 1), padding="same", activation="relu")(creator_conv_layer_3)) # 16
    creator_conv_layer_5_output = layers.BatchNormalization()(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(creator_conv_layer_4)) # 16

    generator_model = keras.Model(generator_input_layer, generator_conv_layer_5_output)
    discriminator_model = keras.Model([discriminator_input_image], [discriminator_hidden_view,
                                                                    discriminator_objects_to_add,
                                                                    discriminator_object_to_sub])
    creator_model = keras.Model(creator_input_hidden_view, creator_conv_layer_5_output)

    return generator_model, discriminator_model, creator_model


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

count_discriminator_first_epochs = 1

min_gen_loss = 0.28
max_gen_loss = 1.0


def fit_generator_and_discriminator_and_creator(generator, discriminator, creator, optimizers, train_x, train_y, epochs, verbose=1, global_num_epoch=1):
    history = []
    global f_save_images
    global count_generator_epochs
    global min_gen_loss
    global max_gen_loss

    if global_num_epoch == 0:
        count_generator_epochs = 0

    for e in range(epochs):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as cr_tape:
            generated_images = generator(train_x, training=True)

            hidden_view_fake, objects_to_add_fake, objects_to_sub_fake = discriminator(generated_images, training=True)
            hidden_view_real, objects_to_add_real, object_to_sub_real = discriminator(train_y, training=True)

            restored_fake_image = creator(hidden_view_fake, training=True)
            restored_real_image = creator(hidden_view_real, training=True)

            cr_fake_loss = tf.reduce_mean(mean_squared_error(generated_images, restored_fake_image))
            cr_real_loss = tf.reduce_mean(mean_squared_error(train_y, restored_real_image))
            cr_loss = (cr_fake_loss + cr_real_loss) / 2

            real_from_fake = tf.sigmoid((tf.sigmoid((hidden_view_fake + objects_to_add_fake) * 6) - objects_to_sub_fake) * 6)
            #true, predict

            disc_zero_real_add_loss = tf.reduce_mean(mean_squared_error(tf.zeros_like(objects_to_add_real), objects_to_add_real))
            disc_zero_real_sub_loss = tf.reduce_mean(mean_squared_error(tf.zeros_like(object_to_sub_real), object_to_sub_real))
            disc_real_loss = (disc_zero_real_add_loss + disc_zero_real_sub_loss) / 2

            disc_max_differences_between_fake_and_real_loss = -tf.reduce_mean(mean_squared_error(hidden_view_real, hidden_view_fake))
            disc_fake_add_sub_loss = tf.reduce_mean(mean_squared_error(hidden_view_real, real_from_fake))
            disc_add_loss = tf.reduce_mean(objects_to_add_fake)
            disc_sub_loss = tf.reduce_mean(objects_to_sub_fake)
            disc_loss = (cr_fake_loss * 2 + cr_real_loss * 2 + disc_fake_add_sub_loss * 2 + disc_max_differences_between_fake_and_real_loss + disc_real_loss + disc_add_loss * 1 + disc_sub_loss * (-1)) / 7

            gen_zero_fake_add_loss = tf.reduce_mean(mean_squared_error(tf.zeros_like(objects_to_add_fake), objects_to_add_fake))
            gen_zero_fake_sub_loss = tf.reduce_mean(mean_squared_error(tf.zeros_like(objects_to_sub_fake), objects_to_sub_fake))
            gen_loss = (gen_zero_fake_add_loss + gen_zero_fake_sub_loss) / 2

            gan_loss = tf.reduce_mean(tf.losses.mean_squared_error(train_y, generated_images))

        #generator_gan_gradient = gen_tape.gradient(gan_loss, generator.trainable_variables)

        # show_image_tensor(tf.expand_dims(generated_images[0], axis=0))

        generator_optimizer, discriminator_optimizer, creator_optimizer = optimizers

        if gen_loss >= min_gen_loss:
            generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))

            if global_num_epoch % f_save_images == 0:
                show_image_tensor(tf.expand_dims(generated_images[0], axis=0), path_to_save_image=f"results\image_epoch_{global_num_epoch}.png", to_show_image=False)
            count_generator_epochs += 1

        if gen_loss < max_gen_loss:
            discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            creator_gradient = cr_tape.gradient(cr_loss, creator.trainable_variables)

            discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
            creator_optimizer.apply_gradients(zip(creator_gradient, creator.trainable_variables))


        #generator_optimizer.apply_gradients(zip(generator_gan_gradient, generator.trainable_variables))

        if verbose == 1:
            print(f"Epoch {epochs}: gen_loss = {gen_loss}, disc_loss = {disc_loss}, cr_loss = {cr_loss}")

        history.append([gan_loss, gen_loss, disc_loss, cr_loss])
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
    creator = None
else:
    generator, discriminator, creator = get_generator_and_discriminator_and_creator(count_input_neurons)

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

    generator_optimizer = keras.optimizers.Adam(learning_rate=0.00004)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.00004)
    creator_optimizer = keras.optimizers.Adam(learning_rate=0.00004)

    optimizers = (generator_optimizer, discriminator_optimizer, creator_optimizer)

    epoch = 0
    while True:
        history = fit_generator_and_discriminator_and_creator(generator, discriminator, creator, optimizers, random_input, images, 1, verbose=0, global_num_epoch=epoch)

        print(f"epoch {epoch}: gan_loss = {history[0][0]}, gen_loss = {history[0][1]}, disc_loss = {history[0][2]}, cr_loss = {history[0][3]}")

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