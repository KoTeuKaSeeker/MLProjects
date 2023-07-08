import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

with open('data/karpitancky_dochka.txt', 'r', encoding='utf-8') as f:
    text1 = f.read()
    text1 = text1.replace('\ufeff', '')
    text1 = re.sub(r'[^А-я ]', '', text1)

with open('data/otcy_i_dety.txt', 'r', encoding='utf-8') as f:
    text2 = f.read()
    text2 = text2.replace('\ufeff', '')
    text2 = re.sub(r'[^А-я ]', '', text2)

text = text1 + text2

max_words_count = 5000
tokenizer = Tokenizer(num_words=max_words_count, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)

tokenizer.fit_on_texts([text])

sequence_of_text1 = tokenizer.texts_to_sequences([text1])[0]
sequence_of_text2 = tokenizer.texts_to_sequences([text2])[0]

count_words_in_train = 10
count_words_in_text1 = len(sequence_of_text1) - count_words_in_train
count_words_in_text2 = len(sequence_of_text2) - count_words_in_train
count_samples = count_words_in_text1 + count_words_in_text2

X = [sequence_of_text1[i:i + count_words_in_train] for i in range(count_words_in_text1)]
X += [sequence_of_text2[i:i + count_words_in_train] for i in range(count_words_in_text2)]
Y = [[1, 0]] * count_words_in_text1 + [[0, 1]] * count_words_in_text2

X = np.array(X)
Y = np.array(Y)

indeces = np.random.choice(len(X), size=len(X), replace=False)
X = X[indeces]
Y = Y[indeces]

model = keras.Sequential()
model.add(layers.Embedding(max_words_count, 128, input_shape=(None,)))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(64))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=2)


def predict_number_of_text(input_str):
    data = tokenizer.texts_to_sequences([input_str])

    if len(data[0]) == 0:
        return -1

    prediction = model.predict(data)
    return prediction.argmax(axis=1)


while True:
    chunk = input('Enter a chunk of text: ')
    print('The chunk of text from a text ' + str(predict_number_of_text(chunk)))
    print('')