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


def get_generator(count_input_neurons):
    # Generator
    ###########################################################
    generator_input_layer = layers.Input(shape=count_input_neurons)

    generator_dense_layer = layers.BatchNormalization()(layers.Dense(units=4 * 4 * 20, activation="sigmoid")(generator_input_layer))
    generator_dense_reshape_layer = layers.Reshape((4, 4, 20))(generator_dense_layer)

    generator_conv_layer_1 = layers.BatchNormalization()(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", activation="sigmoid")(generator_dense_reshape_layer)) # 8
    generator_conv_layer_2 = layers.BatchNormalization()(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", activation="sigmoid")(generator_conv_layer_1)) # 16
    generator_conv_layer_3 = layers.BatchNormalization()(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(generator_conv_layer_2)) # 32
    generator_conv_layer_4 = layers.BatchNormalization()(layers.Conv2DTranspose(48, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(generator_conv_layer_3)) # 32
    generator_conv_layer_5_output = layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding="same", activation="sigmoid")(generator_conv_layer_4) # 64

    generator_model = keras.Model(generator_input_layer, generator_conv_layer_5_output)

    image_shape = (generator_conv_layer_5_output.shape[1],
                   generator_conv_layer_5_output.shape[2],
                   generator_conv_layer_5_output.shape[3])

    return generator_model, image_shape


def get_discriminator(count_input_neurons, image_shape):
    # Discriminator
    ###########################################################

    discriminator_input_image = layers.Input(shape=image_shape)
    discriminator_input_labels = layers.Input(shape=count_input_neurons)

    discriminator_conv_layer_1 = layers.Conv2D(4, (5, 5), strides=(1, 1), padding="same", activation="relu")(discriminator_input_image) # 16
    discriminator_conv_layer_2 = layers.Conv2D(5, (5, 5), strides=(1, 1), padding="same", activation="relu")(discriminator_conv_layer_1) # 16
    discriminator_conv_layer_3 = layers.Conv2D(8, (5, 5), strides=(2, 2), padding="same", activation="relu")(discriminator_conv_layer_2) # 8
    discriminator_conv_layer_4 = layers.Conv2D(10, (5, 5), strides=(2, 2), padding="same", activation="relu")(discriminator_conv_layer_3) # 4
    discriminator_conv_layer_5_output = layers.Conv2D(12, (5, 5), strides=(2, 2), padding="same", activation="relu")(discriminator_conv_layer_4) # 4
    discriminator_flatten = layers.Flatten()(discriminator_conv_layer_5_output)

    discriminator_input_labels_flatten = layers.Flatten()(discriminator_input_labels)
    discriminator_dense_input = layers.Concatenate()([discriminator_flatten, discriminator_input_labels_flatten])

    discriminator_dense_1 = layers.Dense(units=64, activation="relu")(discriminator_dense_input)
    discriminator_dense_2 = layers.Dense(units=32, activation="relu")(discriminator_dense_1)
    discriminator_dense_3 = layers.Dense(units=16, activation="relu")(discriminator_dense_2)
    discriminator_dense_4_output = layers.Dense(units=1)(discriminator_dense_3)

    ###########################################################

    discriminator_model = keras.Model([discriminator_input_image, discriminator_input_labels],
                                      discriminator_dense_4_output)

    return discriminator_model


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


def discriminators_predict(discriminators, y_data, x_data, training=False):
    prediction = 0

    for discriminator_id, discriminator in enumerate(discriminators):
        prediction += discriminator([y_data, x_data], training=(training if discriminator_id == len(discriminators) - 1 else False))

    return tf.sigmoid(prediction)


f_save_images = 10
count_not_changed_min_max_to_add_discriminator = 200

gan_loss_max = -99999
gan_loss_min = 999999
count_not_changed_min_max = 0
count_generator_epochs = 0

discriminator_optimizer_for_fit = None


def fit_generator_and_discriminator(generator: keras.Model, generator_optimizer, discriminators, discriminator_learning_rate, train_x, train_y, epochs, to_train_generator_previous, verbose=1, global_num_epoch=1):
    history = []
    to_train_generator = to_train_generator_previous
    min_loss_to_swap_training = 0.9
    keeping_fake_value = 0.5
    global f_save_images
    global count_not_changed_min_max_to_add_discriminator
    global gan_loss_max
    global gan_loss_min
    global count_not_changed_min_max
    global count_generator_epochs
    global discriminator_optimizer_for_fit

    if global_num_epoch == 0:
        gan_loss_max = -99999
        gan_loss_min = 999999
        count_not_changed_min_max = 0
        count_generator_epochs = 0
        discriminator_optimizer_for_fit = keras.optimizers.Adam(learning_rate=discriminator_learning_rate)

    for e in range(epochs):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(train_x, training=True)

            d_fake = discriminators_predict(discriminators, generated_images, train_x, training=True)
            d_real = discriminators_predict(discriminators, train_y, train_x, training=True)

            d_fake_mean = tf.reduce_mean(d_fake)

            gen_loss = tf.reduce_mean(cross_entropy(tf.zeros_like(d_fake), d_fake))

            disc_real_loss = tf.reduce_mean(cross_entropy(tf.zeros_like(d_real), d_real))
            disc_fake_loss = tf.reduce_mean(cross_entropy(tf.ones_like(d_fake), d_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            gen_loss_norm = gen_loss
            disc_loss_norm = disc_loss

            gan_loss = tf.reduce_mean(tf.losses.mean_squared_error(train_y, generated_images))

        #generator_gan_gradient = gen_tape.gradient(gan_loss, generator.trainable_variables)

        # show_image_tensor(tf.expand_dims(generated_images[0], axis=0))

        if d_fake_mean > keeping_fake_value:
            generator_gradient = gen_tape.gradient(gen_loss_norm, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))

            if global_num_epoch % f_save_images == 0:
                show_image_tensor(tf.expand_dims(generated_images[0], axis=0), path_to_save_image=f"results\image_epoch_{global_num_epoch}.png", to_show_image=False)

            count_generator_epochs += 1

        if d_fake_mean <= keeping_fake_value:
            discriminator_gradient = disc_tape.gradient(disc_loss_norm, discriminators[-1].trainable_variables)
            discriminator_optimizer_for_fit.apply_gradients(zip(discriminator_gradient, discriminators[-1].trainable_variables))

        if gan_loss > gan_loss_max:
            gan_loss_max = gan_loss
            count_not_changed_min_max = 0

        if gan_loss < gan_loss_min:
            gan_loss_min = gan_loss
            count_not_changed_min_max = 0

        count_not_changed_min_max += 1

        if count_not_changed_min_max >= count_not_changed_min_max_to_add_discriminator:
            discriminator_optimizer_for_fit = keras.optimizers.Adam(learning_rate=discriminator_learning_rate)
            discriminators.append(get_discriminator(train_x.shape[1], (train_y.shape[1], train_y.shape[2], train_y.shape[3])))
            count_not_changed_min_max = 0
            gan_loss_max = -99999
            gan_loss_min = 999999


        #generator_optimizer.apply_gradients(zip(generator_gan_gradient, generator.trainable_variables))

        if verbose == 1:
            print(f"Epoch {epochs}: gen_loss = {gen_loss}, disc_loss = {disc_loss}")

        history.append([gan_loss, gen_loss, disc_loss])
    return history, to_train_generator


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
images = tf.stack(tf.unstack(images, axis=0)[:150])
random_input = tf.random.uniform((images.shape[0], count_input_neurons), minval=0, maxval=1)

to_load_model = input("Do you need load model from file? (y/n)") == "y"
if to_load_model:
    path_to_model = input("Enter the path to the model: ")
    generator, discriminator = load_generator_and_discriminator(path_to_model)
else:
    generator, image_shape = get_generator(count_input_neurons)
    discriminator = get_discriminator(count_input_neurons, image_shape)

discriminators = [discriminator]

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

    generator_optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    discriminator_optimizers = [keras.optimizers.Adam(learning_rate=0.00001)]

    to_train_generator = False
    epoch = 0
    while True:
        history, to_train_generator = fit_generator_and_discriminator(generator, generator_optimizer, discriminators, 0.00001, random_input, images, 1, to_train_generator, verbose=0, global_num_epoch=epoch)



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
                #save_generator_and_discriminator(generator, discriminator, path_to_save_model)

        epoch += 1

    to_save_model = input("Do you want to save model? (y/n)") == "y"
    if to_save_model:
        path_to_save_model = input("Enter the path to save the model: ")
        #save_generator_and_discriminator(generator, discriminator, path_to_save_model)

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