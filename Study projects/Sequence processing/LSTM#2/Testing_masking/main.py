import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.sequence import pad_sequences


def get_input_data(count_samples, max_count_time_steps, input_dim, masking_value):
    testing_data = []
    input_data = np.full((count_samples, max_count_time_steps, input_dim), masking_value).tolist()
    for sample in range(count_samples):
        count_time_steps = random.randint(1, max_count_time_steps)
        sample_without_padding = [[[]] * count_time_steps]
        for time_step in range(count_time_steps):
            input_data[sample][time_step] = np.random.rand(input_dim).tolist()
            sample_without_padding[0][time_step] = input_data[sample][time_step]
        testing_data.append(tf.constant(sample_without_padding))
    input_data = tf.constant(input_data)
    return input_data, testing_data


def testing_masking(model:keras.Model, input_data):
    masked_input_data = model.get_layer("masking_layer")(input_data)
    print('kek')

input_dim = 1
lstm_dim = 20
dense_dim = 1

masking_value = 0
max_count_time_steps = 15
count_samples = 4

input_data, testing_data = get_input_data(count_samples, max_count_time_steps, input_dim, masking_value)
output_data = tf.constant(np.random.rand(count_samples, dense_dim))

input_layer = layers.Input(shape=(None, input_dim))
masking_layer = layers.Masking(mask_value=masking_value, name="masking_layer")(input_layer)
lstm_layer = layers.LSTM(units=lstm_dim, input_shape=(None,))(masking_layer)
dense_layer = layers.Dense(units=dense_dim, activation="sigmoid")(lstm_layer)

model = keras.Model(inputs=input_layer, outputs=dense_layer)

testing_masking(model, input_data)

model.compile(optimizer="adam", loss="mse")
model.fit(input_data, output_data, batch_size=32, epochs=500)

################################################################################

print('\nTesting masking:')

for sample_id in range(count_samples):
    sample_target = tf.reshape(output_data[sample_id], (1, dense_dim))
    sample_with_padding = tf.reshape(input_data[sample_id], (1, max_count_time_steps, input_dim))
    print(f'Loss with padding: {model.evaluate(sample_with_padding, sample_target, verbose=0)}')
    print(f'Loss without padding: {model.evaluate(testing_data[sample_id], sample_target, verbose=0)}\n')
