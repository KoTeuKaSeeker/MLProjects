import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import numpy as np
import math
import enum

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.text import Tokenizer

test_data = tf.zeros((1, 1))

###################################################################

model0_input = layers.Input(shape=(None, 1), name='model0_input')

model0_lstm = layers.LSTM(units=10, input_shape=(None,), return_state=True, name='model0_lstm')
model0_lstm_output, model0_state_h, model0_state_c = model0_lstm(model0_input)
model0_states = [model0_state_h, model0_state_c]

model0 = keras.Model(inputs=model0_input, outputs=model0_states)
print(model0.predict(test_data))

###################################################################

model1_input_h = layers.Input(shape=(10,))
model1_input_c = layers.Input(shape=(10,))
model1_input_states = [model1_input_h, model1_input_c]

model1_input_output = model0.get_layer('model0_input').output

model1_lstm_output, model1_state_h, model1_state_c = model0.get_layer('model0_lstm')(model1_input_output, initial_state=model1_input_states)
model1_states = [model1_state_h, model1_state_c]

model1 = keras.Model(inputs=[model1_input_output, model0_states], outputs=model1_lstm_output)
print(model1.predict([test_data, [tf.zeros(1, 10), tf.zeros(1, 10)]]))