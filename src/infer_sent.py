from __future__ import print_function

import preprocessing as preprocessing
import model_utils as model_utils
from keras.models import Model
from keras.layers import TimeDistributed, Embedding, Lambda, Dense, concatenate, BatchNormalization, Bidirectional, \
    LSTM, Dropout, GaussianNoise, GlobalMaxPooling1D, MaxPooling1D
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import History

history = History()
n_hidden = 512


def create_model(model_input, word_embedding_matrix, maxlen=30, embeddings='glove', sent_embed='univ_sent', lr=1e-3):
    # The visible layer
    if embeddings != 'elmo':
        in_dim, out_dim = word_embedding_matrix.shape
        embedding_layer = Embedding(in_dim,
                                    out_dim,
                                    weights=[word_embedding_matrix],
                                    input_length=maxlen,
                                    trainable=False)
        # Embedded version of the inputs
        encoded_q1 = embedding_layer(model_input[0])
        encoded_q2 = embedding_layer(model_input[1])
    else:
        encoded_q1 = Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(model_input[0])
        encoded_q2 = Lambda(model_utils.ElmoEmbedding, output_shape=(maxlen, 1024))(model_input[1])

    q1_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(model_input[2])
    q2_embed_sent = Lambda(model_utils.UniversalEmbedding, output_shape=(512,))(model_input[3])
    sent1_dense = Dense(256, activation='relu')(q1_embed_sent)
    sent2_dense = Dense(256, activation='relu')(q2_embed_sent)
    distance_sent = Lambda(preprocessing.cosine_distance, output_shape=preprocessing.get_shape)(
        [sent1_dense, sent2_dense])

    shared_layer = Bidirectional(LSTM(n_hidden, kernel_initializer='glorot_uniform', return_sequences=True))
    max_pool = GlobalMaxPooling1D()

    output_q1 = shared_layer(encoded_q1)
    output_q2 = shared_layer(encoded_q2)

    output_q1 = max_pool(output_q1)
    output_q2 = max_pool(output_q2)

    # output_q1 = GaussianNoise(0.01)(output_q1)
    # output_q2 = GaussianNoise(0.01)(output_q2)

    abs_diff = Lambda(preprocessing.abs_difference, output_shape=preprocessing.get_shape)(
        [output_q1, output_q2])
    mult = Lambda(preprocessing.multiplication, output_shape=preprocessing.get_shape)([output_q1, output_q2])

    if sent_embed == 'univ_sent':
        merged = concatenate([output_q1, output_q2, abs_diff, mult, distance_sent])
    else:
        merged = concatenate([output_q1, output_q2, abs_diff, mult])

    merged = Dense(n_hidden, activation='relu')(merged)
    merged = Dropout(0.1)(merged)
    merged = BatchNormalization()(merged)

    output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001),
                   bias_regularizer=regularizers.l2(0.0001))(merged)

    # Pack it all up into a model
    net = Model(model_input, [output])

    net.compile(optimizer=SGD(lr=lr), loss='binary_crossentropy',
                metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
    return net
