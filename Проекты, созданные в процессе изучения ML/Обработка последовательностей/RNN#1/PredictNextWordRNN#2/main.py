import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical

with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')

max_word_count = 1000
tokenizer = Tokenizer(num_words=max_word_count, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([text])

data = np.array(tokenizer.texts_to_sequences([text]))
res = data[0]

inp_words = 3
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words] for i in range(n)])
Y = keras.utils.to_categorical(res[inp_words:], num_classes=max_word_count)

model = keras.Sequential()
model.add(layers.Embedding(max_word_count, 256, input_length=inp_words))
model.add(layers.SimpleRNN(128, activation='tanh'))
model.add(layers.Dense(max_word_count, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=50)


def build_phrase(input_text, count_predictions=20):
    data = tokenizer.texts_to_sequences([input_text])[0]
    for prediction_id in range(count_predictions):
        print(data)
        X = data[len(data) - inp_words:]
        X = np.expand_dims(X, axis=0)

        print(X.shape)

        pred = model.predict(X)
        index = pred.argmax(axis=1)[0]
        data.append(index)

        input_text += " " + tokenizer.index_word[index]
    return input_text

while True:
    input_text = input("Enter a input text: ")
    print("RNN's answer: " + build_phrase(input_text))
    print()