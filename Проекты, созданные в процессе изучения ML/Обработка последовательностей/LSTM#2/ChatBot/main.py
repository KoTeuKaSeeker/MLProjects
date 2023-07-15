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

# decoder_encoder_chat_bot = keras.Model(inputs=[encoder_word_input,
#                                                encoder_word_index_input,
#                                                encoder_reply_index_input,
#                                                encoder_person_index_input,
#                                                decoder_input],
#                                        outputs=decoder_dense_output)

class EncoderState:
    encoder_hidden_states = []
    word_id = 0
    reply_id = 0
    person_id = 0

def dec_to_bin_array(size_array, num):
    character_bit_array = list(bin(num)[2:])
    character_bit_array = [int(bit) for bit in character_bit_array]
    character_bit_array = [0] * (size_array - len(character_bit_array)) + character_bit_array

    return character_bit_array


def convert_to_encoder_data(text, encoder_state, tokenizer):
    words = tokenizer.texts_to_sequences([text])[0]
    word_indices = tf.reshape(tf.constant(np.arange(encoder_state.word_id, encoder_state.word_id + len(words))), (1, -1, 1))
    reply_indices = tf.constant([[[encoder_state.reply_id]] * len(words)])
    person_indices = tf.constant([[[(encoder_state.person_id + person_id ) % 2] for person_id in range(len(words))]])
    words = tf.reshape(tf.constant(words), (1, -1, 1))
    encoder_state.reply_id += 1
    encoder_state.person_id = (encoder_state.person_id + 1) % 2
    return words, word_indices, reply_indices, person_indices


with open('data/conversationRu.txt', 'r', encoding='utf-8') as f:
    raw_dialogues = f.read()

questions_replies = raw_dialogues.split('\n')
questions_replies = [i.split('\t') for i in questions_replies]

dialogues = []

pointer = 0
for i in range(1, len(questions_replies)):
    if questions_replies[i][0][:-1] != questions_replies[i - 1][1]:
        dialogues.append([qa[0] for qa in questions_replies[pointer:i]])
        pointer = i

dialogues.append([qa[0] for qa in questions_replies[pointer:len(questions_replies)]])

max_word_count = 5000
tokenizer = Tokenizer(num_words=max_word_count, filters='–"—#$%&amp;()*+/;<=>@[\\]^_`{|}~\t\n\r«»', lower=True,
                      split=' ', char_level=False)
tokenizer.fit_on_texts([raw_dialogues])

encoder_vocab_size = 6000
encoder_max_value_word_index = 3000
encoder_max_reply_index = 100

count_specialized_tokens = 2
decoder_vocab_size = 6000 + count_specialized_tokens

start_token = decoder_vocab_size - count_specialized_tokens + 1
end_token = decoder_vocab_size - count_specialized_tokens + 2

decoder_output_size = math.ceil(math.log2(decoder_vocab_size)) + 1

start_token_in_binary = dec_to_bin_array(decoder_output_size, start_token)
end_token_in_binary = dec_to_bin_array(decoder_output_size, end_token)

dialogues_in_sequences = []
for i in range(len(dialogues)):
    dialogues_in_sequences.append(tokenizer.texts_to_sequences(dialogues[i]))

words_train_input = []
word_index_train_input = []
reply_index_train_input = []
person_id_train_input = []

decoder_words_train_input = []
decoder_words_train_output = []

max_count_encoder_time_steps = 0
max_count_decoder_time_steps = 0
temp_sample_id = 0
for dialogue_id in range(len(dialogues_in_sequences)):
    for count_replies_in_sample in range(1, len(dialogues_in_sequences[dialogue_id])):
        general_word_id = 0
        words_train_input.append([])
        word_index_train_input.append([])
        reply_index_train_input.append([])
        person_id_train_input.append([])
        decoder_words_train_output.append([])
        decoder_words_train_input.append([])
        for reply_id in range(count_replies_in_sample):
            for word_id in range(len(dialogues_in_sequences[dialogue_id][reply_id])):
                words_train_input[temp_sample_id].append([dialogues_in_sequences[dialogue_id][reply_id][word_id]])
                word_index_train_input[temp_sample_id].append([general_word_id])
                reply_index_train_input[temp_sample_id].append([reply_id])
                person_id_train_input[temp_sample_id].append([reply_id % 2])
                general_word_id += 1

        max_count_encoder_time_steps = max(max_count_encoder_time_steps, len(words_train_input[temp_sample_id]))

        decoder_words_train_input[temp_sample_id].append(start_token_in_binary)
        for word_id in range(len(dialogues_in_sequences[dialogue_id][count_replies_in_sample])):
            word_token_in_binary = dec_to_bin_array(decoder_output_size, dialogues_in_sequences[dialogue_id][count_replies_in_sample][word_id])
            decoder_words_train_input[temp_sample_id].append(word_token_in_binary)
            decoder_words_train_output[temp_sample_id].append(word_token_in_binary)
        decoder_words_train_output[temp_sample_id].append(end_token_in_binary)

        max_count_decoder_time_steps = max(max_count_decoder_time_steps, len(decoder_words_train_input[temp_sample_id]))

        temp_sample_id += 1

count_samples = temp_sample_id

for sample_id in range(count_samples):
    words_train_input[sample_id] = words_train_input[sample_id] + [[0]] * (max_count_encoder_time_steps - len(words_train_input[sample_id]))
    word_index_train_input[sample_id] = word_index_train_input[sample_id] + [[0]] * (max_count_encoder_time_steps - len(word_index_train_input[sample_id]))
    reply_index_train_input[sample_id] = reply_index_train_input[sample_id] + [[0]] * (max_count_encoder_time_steps - len(reply_index_train_input[sample_id]))
    person_id_train_input[sample_id] = person_id_train_input[sample_id] + [[0]] * (max_count_encoder_time_steps - len(person_id_train_input[sample_id]))

    decoder_words_train_input[sample_id] = decoder_words_train_input[sample_id] + [end_token_in_binary] * (max_count_decoder_time_steps - len(decoder_words_train_input[sample_id]))
    decoder_words_train_output[sample_id] = decoder_words_train_output[sample_id] + [end_token_in_binary] * (max_count_decoder_time_steps - len(decoder_words_train_output[sample_id]))

words_train_input_tensor = tf.constant(words_train_input)
word_index_train_input_tensor = tf.constant(word_index_train_input)
reply_index_train_input_tensor = tf.constant(reply_index_train_input)
person_id_train_input_tensor = tf.constant(person_id_train_input)

decoder_words_train_input_tensor = tf.constant(decoder_words_train_input)
decoder_words_train_output_tensor = tf.constant(decoder_words_train_output)


encoder_word_input = layers.Input(shape=1, name='encoder_word_input')
encoder_word_index_input = layers.Input(shape=1, name='encoder_word_index_input')
encoder_reply_index_input = layers.Input(shape=1, name='encoder_reply_index_input')
encoder_person_index_input = layers.Input(shape=(None, 1), name='encoder_person_index_input')

encoder_word_embedded = layers.Embedding(input_dim=encoder_vocab_size, output_dim=64, input_shape=(None,), name='encoder_word_embedded')
encoder_word_index_embedded = layers.Embedding(input_dim=encoder_max_value_word_index, output_dim=64, input_shape=(None,), name='encoder_word_index_embedded')
encoder_reply_index_embedded = layers.Embedding(input_dim=encoder_max_reply_index, output_dim=16, input_shape=(None,), name='encoder_reply_index_embedded')

encoder_word_embedded_output = encoder_word_embedded(encoder_word_input)
encoder_word_index_embedded_output = encoder_word_index_embedded(encoder_word_index_input)
encoder_reply_index_embedded_output = encoder_reply_index_embedded(encoder_reply_index_input)

merged = layers.Concatenate()([encoder_word_embedded_output,
                               encoder_word_index_embedded_output,
                               encoder_reply_index_embedded_output,
                               encoder_person_index_input])

encoder_lstm_1 = layers.LSTM(units=300, input_shape=(None,), return_sequences=True, return_state=True, name='encoder_lstm_1')
encoder_lstm_2 = layers.LSTM(units=150, input_shape=(None,), return_sequences=True, return_state=True, name='encoder_lstm_2')
encoder_lstm_3 = layers.LSTM(units=100, input_shape=(None,), return_state=True, name='encoder_lstm_3')

encoder_lstm_output_1, encoder_state_h_1, encoder_state_c_1 = encoder_lstm_1(merged)
encoder_lstm_output_2, encoder_state_h_2, encoder_state_c_2 = encoder_lstm_2(encoder_lstm_output_1)
encoder_lstm_output_3, encoder_state_h_3, encoder_state_c_3 = encoder_lstm_3(encoder_lstm_output_2)

encoder_states_1 = [encoder_state_h_1, encoder_state_c_1]
encoder_states_2 = [encoder_state_h_2, encoder_state_c_2]
encoder_states_3 = [encoder_state_h_3, encoder_state_c_3]
encoder_states = [encoder_states_1, encoder_states_2, encoder_states_3]

decoder_input = layers.Input(shape=(None, decoder_output_size), name='decoder_input')

decoder_lstm_1 = layers.LSTM(units=300, input_shape=(None,), return_sequences=True, return_state=True, name='decoder_lstm_1')
decoder_lstm_2 = layers.LSTM(units=150, input_shape=(None,), return_sequences=True, return_state=True, name='decoder_lstm_2')
decoder_lstm_3 = layers.LSTM(units=100, input_shape=(None,), return_sequences=True, return_state=True, name='decoder_lstm_3')
decoder_dense = layers.Dense(units=decoder_output_size, activation='sigmoid', name='decoder_dense')

decoder_lstm_output_1, decoder_state_h_1, decoder_state_c_1 = decoder_lstm_1(decoder_input, initial_state=encoder_states_1)
decoder_lstm_output_2, decoder_state_h_2, decoder_state_c_2 = decoder_lstm_2(decoder_lstm_output_1, initial_state=encoder_states_2)
decoder_lstm_output_3, decoder_state_h_3, decoder_state_c_3 = decoder_lstm_3(decoder_lstm_output_2, initial_state=encoder_states_3)
decoder_dense_output = decoder_dense(decoder_lstm_output_3)

decoder_states_1 = [decoder_state_h_1, decoder_state_c_1]
decoder_states_2 = [decoder_state_h_2, decoder_state_c_2]
decoder_states_3 = [decoder_state_h_3, decoder_state_c_3]
decoder_states = [decoder_states_1, decoder_states_2, decoder_states_3]

decoder_encoder_chat_bot = keras.Model(inputs=[encoder_word_input,
                                               encoder_word_index_input,
                                               encoder_reply_index_input,
                                               encoder_person_index_input,
                                               decoder_input],
                                       outputs=decoder_dense_output)
decoder_encoder_chat_bot.summary()

decoder_encoder_chat_bot.compile(optimizer="adam", loss='mse')

decoder_encoder_chat_bot.fit([words_train_input_tensor, word_index_train_input_tensor, reply_index_train_input_tensor, person_id_train_input_tensor, decoder_words_train_input_tensor], decoder_words_train_output_tensor,
                            batch_size=32, epochs=1000)

#####################################################################################

r_encoder_input_state_h_1 = layers.Input(shape=(300,))
r_encoder_input_state_c_1 = layers.Input(shape=(300,))

r_encoder_input_state_h_2 = layers.Input(shape=(150,))
r_encoder_input_state_c_2 = layers.Input(shape=(150,))

r_encoder_input_state_h_3 = layers.Input(shape=(100,))
r_encoder_input_state_c_3 = layers.Input(shape=(100,))

encoder_input_states_1 = [r_encoder_input_state_h_1, r_encoder_input_state_c_1]
encoder_input_states_2 = [r_encoder_input_state_h_2, r_encoder_input_state_c_2]
encoder_input_states_3 = [r_encoder_input_state_h_3, r_encoder_input_state_c_3]
encoder_input_states = [encoder_input_states_1, encoder_input_states_2, encoder_input_states_3]

r_encoder_lstm_output_1, r_encoder_state_h_1, r_encoder_state_c_1 = encoder_lstm_1(merged, initial_state=encoder_input_states_1)
r_encoder_lstm_output_2, r_encoder_state_h_2, r_encoder_state_c_2 = encoder_lstm_2(encoder_lstm_output_1, initial_state=encoder_input_states_2)
r_encoder_lstm_output_3, r_encoder_state_h_3, r_encoder_state_c_3 = encoder_lstm_3(encoder_lstm_output_2, initial_state=encoder_input_states_3)

r_encoder_states_1 = [r_encoder_state_h_1, r_encoder_state_c_1]
r_encoder_states_2 = [r_encoder_state_h_2, r_encoder_state_c_2]
r_encoder_states_3 = [r_encoder_state_h_3, r_encoder_state_c_3]
r_encoder_states = [r_encoder_states_1, r_encoder_states_2, r_encoder_states_3]

chat_bot_encoder = keras.Model(inputs=[encoder_word_input,
                                       encoder_word_index_input,
                                       encoder_reply_index_input,
                                       encoder_person_index_input,
                                       encoder_input_states],
                               outputs=encoder_states)

r_decoder_input_state_h_1 = layers.Input(shape=(300,))
r_decoder_input_state_c_1 = layers.Input(shape=(300,))

r_decoder_input_state_h_2 = layers.Input(shape=(150,))
r_decoder_input_state_c_2 = layers.Input(shape=(150,))

r_decoder_input_state_h_3 = layers.Input(shape=(100,))
r_decoder_input_state_c_3 = layers.Input(shape=(100,))

r_decoder_input_states_1 = [r_decoder_input_state_h_1, r_decoder_input_state_c_1]
r_decoder_input_states_2 = [r_decoder_input_state_h_2, r_decoder_input_state_c_2]
r_decoder_input_states_3 = [r_decoder_input_state_h_3, r_decoder_input_state_c_3]
r_decoder_input_states = [r_decoder_input_states_1, r_decoder_input_states_2, r_decoder_input_states_3]

r_decoder_lstm_output_1, r_decoder_state_h_1, r_decoder_state_c_1 = decoder_lstm_1(decoder_input, initial_state=r_decoder_input_states_1)
r_decoder_lstm_output_2, r_decoder_state_h_2, r_decoder_state_c_2 = decoder_lstm_2(r_decoder_lstm_output_1, initial_state=r_decoder_input_states_2)
r_decoder_lstm_output_3, r_decoder_state_h_3, r_decoder_state_c_3 = decoder_lstm_3(r_decoder_lstm_output_2, initial_state=r_decoder_input_states_3)
r_decoder_dense_output = decoder_dense(r_decoder_lstm_output_3)

r_decoder_states_1 = [r_decoder_state_h_1, r_decoder_state_c_1]
r_decoder_states_2 = [r_decoder_state_h_2, r_decoder_state_c_2]
r_decoder_states_3 = [r_decoder_state_h_3, r_decoder_state_c_3]
r_decoder_states = [r_decoder_states_1, r_decoder_states_2, r_decoder_states_3]

chat_bot_decoder = keras.Model(inputs=[decoder_input, r_decoder_input_states], outputs=[r_decoder_dense_output, r_decoder_states])


def get_answer(prompt, encoder_state_container, max_answer_length=100):
    words, word_indices, reply_indices, person_indices = convert_to_encoder_data(prompt, encoder_state_container, tokenizer)
    encoder_states = chat_bot_encoder.predict([words, word_indices, reply_indices, person_indices, encoder_state_container.encoder_hidden_states], verbose=0)

    encoder_state_container.encoder_hidden_states = encoder_states

    hidden_states = encoder_states
    input_word = tf.reshape(tf.constant(start_token_in_binary), (1, 1, -1))

    answer = ""

    for time_step in range(max_answer_length):
        prediction, hidden_states = chat_bot_decoder.predict([input_word, hidden_states], verbose=0)
        input_word = prediction

        mask = prediction > 0.5
        binary_tensor = tf.where(mask, tf.ones_like(prediction), tf.zeros_like(prediction))
        token = int(tf.reduce_sum(binary_tensor[0][0] * tf.cast(tf.pow(2, tf.range(tf.size(binary_tensor[0][0]))[::-1]), dtype=tf.float32))) #ОЧЕНЬ ОЧЕНЬ ВЕРОЯТНО, ЧТО ЗДЕСЬ ОШИБКА!!!

        if token == end_token:
            break

        if 0 < token < decoder_vocab_size:
            answer += tokenizer.index_word[token] + ' '

    if len(answer) > 0:
        words, word_indices, reply_indices, person_indices = convert_to_encoder_data(answer, encoder_state_container, tokenizer)
        encoder_states_with_answer = chat_bot_encoder.predict([words, word_indices, reply_indices, person_indices, encoder_state_container.encoder_hidden_states])

        encoder_state_container.encoder_hidden_states = encoder_states_with_answer

    return answer

encoder_state_container = EncoderState()

while True:
    your_statement = input('You: ')
    if your_statement == '/reload_states':
        encoder_state_container = EncoderState()
    chat_bot_statement = get_answer(your_statement, encoder_state_container)
    print(f'ChatBot: {chat_bot_statement}')