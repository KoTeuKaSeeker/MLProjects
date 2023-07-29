import numpy as np
import tensorflow as tf
import math
from tensorflow import keras
from keras import layers
from keras.preprocessing.text import Tokenizer

def dec_to_bin_array(size_array, num):
    character_bit_array = list(bin(num)[2:])
    character_bit_array = [int(bit) for bit in character_bit_array]
    character_bit_array = [0] * (size_array - len(character_bit_array)) + character_bit_array

    return character_bit_array

def read_data(path, max_word_count):
    questions = []
    replies = []
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        lines = text.split('\n')
        for line in lines[:min(4, len(lines))]:
            question, reply = line.split('\t')
            questions.append(question)
            replies.append(reply)
    return text, questions, replies


def get_train_data(questions, replies, start_token, end_token, padding_token, tokenizer: Tokenizer):
    decoder_output_dim = math.ceil(math.log2(tokenizer.num_words)) + 1

    encoder_input_data = tokenizer.texts_to_sequences(questions)
    decoder_input_data = tokenizer.texts_to_sequences(replies)

    encoder_input_data = keras.utils.pad_sequences(encoder_input_data, padding="post", value=padding_token).tolist()
    decoder_input_data = keras.utils.pad_sequences(decoder_input_data, padding="post", value=end_token).tolist()
    decoder_target_data = [[[]] * len(decoder_input_data[0])] * len(decoder_input_data)

    count_samples = len(decoder_input_data)
    count_time_steps = len(decoder_input_data[0])
    for sample_id in range(count_samples):
        for time_step_id in range(count_time_steps):
            decoder_target_data[sample_id][time_step_id] = dec_to_bin_array(decoder_output_dim, decoder_input_data[sample_id][time_step_id])
        decoder_input_data[sample_id] = [start_token] + decoder_input_data[sample_id]
        decoder_target_data[sample_id] = decoder_target_data[sample_id] + [dec_to_bin_array(decoder_output_dim, end_token)]

    encoder_input_data = tf.reshape(tf.constant(encoder_input_data), (len(encoder_input_data), len(encoder_input_data[0]), 1))
    decoder_input_data = tf.reshape(tf.constant(decoder_input_data), (len(decoder_input_data), len(decoder_input_data[0]), 1))
    decoder_target_data = tf.constant(decoder_target_data)

    return encoder_input_data, decoder_input_data, decoder_target_data, decoder_output_dim


def create_encoder_decoder_model(dim_embedding, dim_lstm_1, dim_lstm_2, dim_dense, padding_token, count_tokens):
    encoder_input = layers.Input(shape=1, name="encoder_input")
    encoder_masking = layers.Masking(mask_value=padding_token, input_shape=(None, 1), name="encoder_masking")(encoder_input)
    encoder_embedding = layers.Embedding(input_dim=count_tokens, output_dim=dim_embedding, input_shape=(None,), name="encoder_embedding")(encoder_masking)

    encoder_lstm_1, encoder_state_h_1, encoder_state_c_1 = layers.LSTM(units=dim_lstm_1, input_shape=(None,), return_sequences=True, return_state=True, name="encoder_lstm_1")(encoder_embedding)
    encoder_lstm_2, encoder_state_h_2, encoder_state_c_2 = layers.LSTM(units=dim_lstm_2, input_shape=(None,), return_sequences=True, return_state=True, name="encoder_lstm_2")(encoder_lstm_1)

    encoder_states_1 = [encoder_state_h_1, encoder_state_c_1]
    encoder_states_2 = [encoder_state_h_2, encoder_state_c_2]

    #####################################################################

    decoder_input = layers.Input(shape=1, name="decoder_input")
    decoder_embedding = layers.Embedding(input_dim=count_tokens, output_dim=dim_embedding, input_shape=(None,), name="decoder_embedding")(decoder_input)

    decoder_lstm_1, _, _ = layers.LSTM(units=dim_lstm_1, input_shape=(None,), return_sequences=True, return_state=True, name="decoder_lstm_1")(decoder_embedding, initial_state=encoder_states_1)
    decoder_lstm_2, _, _ = layers.LSTM(units=dim_lstm_2, input_shape=(None,), return_sequences=True, return_state=True, name="decoder_lstm_2")(decoder_lstm_1, initial_state=encoder_states_2)
    decoder_dense = layers.Dense(units=dim_dense, activation="sigmoid", name="decoder_dense")(decoder_lstm_2)

    #####################################################################

    encoder_decoder_model = keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense)
    encoder_decoder_model.compile(optimizer="adam", loss="mse")

    return encoder_decoder_model


def get_encoder(model: keras.Model):
    encoder_input = model.get_layer("encoder_input").output

    _, encoder_state_h_1, encoder_state_c_1 = model.get_layer("encoder_lstm_1").output
    _, encoder_state_h_2, encoder_state_c_2 = model.get_layer("encoder_lstm_2").output

    encoder_states_1 = [encoder_state_h_1, encoder_state_c_1]
    encoder_states_2 = [encoder_state_h_2, encoder_state_c_2]

    encoder = keras.Model(inputs=encoder_input, outputs=[encoder_states_1, encoder_states_2])
    return encoder


def get_decoder(model: keras.Model, dim_lstm_1, dim_lstm_2):
    decoder_input = model.get_layer("decoder_input").output
    decoder_embedding = model.get_layer("decoder_embedding").output

    decoder_input_state_h_1 = layers.Input(shape=dim_lstm_1)
    decoder_input_state_c_1 = layers.Input(shape=dim_lstm_1)
    decoder_input_state_h_2 = layers.Input(shape=dim_lstm_2)
    decoder_input_state_c_2 = layers.Input(shape=dim_lstm_2)

    decoder_input_states_1 = [decoder_input_state_h_1, decoder_input_state_c_1]
    decoder_input_states_2 = [decoder_input_state_h_2, decoder_input_state_c_2]

    decoder_lstm_1, decoder_state_h_1, decoder_state_c_1 = model.get_layer("decoder_lstm_1")(decoder_embedding, initial_state=decoder_input_states_1)
    decoder_lstm_2, decoder_state_h_2, decoder_state_c_2 = model.get_layer("decoder_lstm_2")(decoder_lstm_1, initial_state=decoder_input_states_2)

    decoder_states_1 = [decoder_state_h_1, decoder_state_c_1]
    decoder_states_2 = [decoder_state_h_2, decoder_state_c_2]

    decoder_dense = model.get_layer("decoder_dense")(decoder_lstm_2)

    decoder = keras.Model(inputs=[decoder_input, decoder_input_states_1, decoder_input_states_2], outputs=[decoder_dense, decoder_states_1, decoder_states_2])
    return decoder


def get_answer(question, encoder, decoder, max_answer_length, start_token, end_token, tokenizer, **kwargs):
    input_sequence = tokenizer.texts_to_sequences([question])[0]
    if "sample_input_data" in kwargs:
        input_sequence = kwargs["sample_input_data"]
    input_sequence = tf.reshape(tf.constant(input_sequence), (1, -1, 1))

    hidden_state_1, hidden_state_2 = encoder.predict(input_sequence, verbose=0)

    answer = ""
    input_token = tf.reshape(tf.constant(start_token), (1, 1, 1))
    for time_step_id in range(max_answer_length):
        prediction, hidden_state_1, hidden_state_2 = decoder.predict([input_token, hidden_state_1, hidden_state_2], verbose=0)

        mask = prediction[0][0] > 0.5
        binary_tensor = tf.where(mask, tf.ones_like(prediction[0][0]), tf.zeros_like(prediction[0][0]))
        token = int(tf.reduce_sum(binary_tensor * tf.cast(tf.pow(2, tf.range(tf.size(binary_tensor))[::-1]), dtype=tf.float32)))

        input_token = end_token
        word = ""
        if token in tokenizer.index_word and token != end_token:
            input_token = token
            word = tokenizer.index_word[token]
        answer += word + ' '

        if input_token == end_token:
            break

        input_token = tf.reshape(tf.constant(input_token), (1, 1, 1))

    return answer


def testing_masking(model: keras.Model, encoder_input_data):
    r_encoder_input = encoder_input_data[0]
    r_encoder_masking = model.get_layer("encoder_masking")(r_encoder_input)
    r_encoder_embedding = model.get_layer("encoder_embedding")(r_encoder_masking)

    r_encoder_lstm_1, r_encoder_state_h_1, r_encoder_state_c_1 = model.get_layer("encoder_lstm_1")(r_encoder_embedding)
    r_encoder_lstm_2, r_encoder_state_h_2, r_encoder_state_c_2 = model.get_layer("encoder_lstm_2")(r_encoder_lstm_1)

    r_encoder_states_1 = [r_encoder_state_h_1, r_encoder_state_c_1]
    r_encoder_states_2 = [r_encoder_state_h_2, r_encoder_state_c_2]

    g_encoder_input = encoder_input_data[0]
    g_encoder_embedding = model.get_layer("encoder_embedding")(g_encoder_input)

    g_encoder_lstm_1, g_encoder_state_h_1, g_encoder_state_c_1 = model.get_layer("encoder_lstm_1")(g_encoder_embedding)
    g_encoder_lstm_2, g_encoder_state_h_2, g_encoder_state_c_2 = model.get_layer("encoder_lstm_2")(g_encoder_lstm_1)

    g_encoder_states_1 = [g_encoder_state_h_1, g_encoder_state_c_1]
    g_encoder_states_2 = [g_encoder_state_h_2, g_encoder_state_c_2]

    print('kek')

path_to_dataset = 'data\\conversationRu.txt'

max_word_count_in_tokenizer = 5000
count_tokens = max_word_count_in_tokenizer + 4

start_token = max_word_count_in_tokenizer + 1
end_token = max_word_count_in_tokenizer + 2
padding_token = 0

dim_embedding = 64
dim_lstm_1 = 128
dim_lstm_2 = 128

text, questions, replies = read_data(path_to_dataset, max_word_count_in_tokenizer)

tokenizer = Tokenizer(num_words=max_word_count_in_tokenizer, filters='–"—#$%&amp;()*+/;<=>@[\\]^_`{|}~\t\n\r«»', lower=True,
                      split=' ', char_level=False)
tokenizer.fit_on_texts([text])

encoder_input_data, decoder_input_data, decoder_target_data, decoder_output_dim = get_train_data(questions, replies, start_token, end_token, padding_token, tokenizer)

encoder_decoder_model = create_encoder_decoder_model(dim_embedding, dim_lstm_1, dim_lstm_2, decoder_output_dim, padding_token, count_tokens)
testing_masking(encoder_decoder_model, encoder_input_data)
encoder_decoder_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=1000)

encoder = get_encoder(encoder_decoder_model)
decoder = get_decoder(encoder_decoder_model, dim_lstm_1, dim_lstm_2)

print(get_answer("Привет как дела?", encoder, decoder, 20, start_token, end_token, tokenizer))
print(get_answer("Привет как дела?", encoder, decoder, 20, start_token, end_token, tokenizer, sample_input_data=encoder_input_data[0]))
