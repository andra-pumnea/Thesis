from __future__ import print_function

import model_utils
import preprocessing
from keras.models import Model
from keras.layers import Input, Embedding, GRU, Lambda, Dense, concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import History
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K

sess = tf.Session()
K.set_session(sess)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

batch_size = 32
max_len = 40


def ElmoEmbedding(x):
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(batch_size * [max_len])
    },
        signature="tokens",
        as_dict=True)["elmo"]


history = History()
n_hidden = 250


def create_model(word_embedding_matrix, maxlen=30, embeddings='glove', lr=1e-3):
    # The visible layer
    question1 = Input(shape=(maxlen,), dtype=tf.string)
    question2 = Input(shape=(maxlen,), dtype=tf.string)

    if embeddings == 'glove':
        print(word_embedding_matrix.shape)
        in_dim, out_dim = word_embedding_matrix.shape
        embedding_layer = Embedding(in_dim,
                                    out_dim,
                                    weights=[word_embedding_matrix],
                                    input_length=maxlen,
                                    trainable=True)
        # Embedded version of the inputs
        encoded_q1 = embedding_layer(question1)
        encoded_q2 = embedding_layer(question2)
    else:
        encoded_q1 = Lambda(ElmoEmbedding, output_shape=(maxlen, 1024))(question1)
        encoded_q2 = Lambda(ElmoEmbedding, output_shape=(maxlen, 1024))(question2)

    # Since this is a siamese network, both sides share the same GRU
    shared_layer = GRU(n_hidden, kernel_initializer='glorot_uniform',
                       bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.0001))

    output_q1 = shared_layer(encoded_q1)
    output_q2 = shared_layer(encoded_q2)

    # Calculates the distance as defined by the Euclidean RNN model
    distance = Lambda(preprocessing.exponent_neg_manhattan_distance, output_shape=preprocessing.get_shape)(
        [output_q1, output_q2])

    output = concatenate([output_q1, output_q2, distance])
    output = Dense(1, activation='sigmoid')(output)
    output = BatchNormalization()(output)

    output = Dense(1, activation='sigmoid')(output)

    # Pack it all up into a model
    net = Model([question1, question2], [output])

    net.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                metrics=['binary_crossentropy', 'accuracy', model_utils.f1])
    return net


# net.summary()
def run_gru():
    train_file = 'Quora_question_pair_partition/train.tsv'
    dev_file = 'Quora_question_pair_partition/dev.tsv'
    test_file = 'Quora_question_pair_partition/test.tsv'

    # Prepare datasets
    q1_train, q2_train, y_train, emb_matrix = preprocessing.prepare_dataset(train_file, 1)
    q1_dev, q2_dev, y_dev = preprocessing.prepare_dataset(dev_file)
    q1_test, q2_test, y_test = preprocessing.prepare_dataset(test_file)

    net = create_model(emb_matrix)
    # Start training
    hist = net.fit([q1_train, q2_train], y_train,
                   validation_data=([q1_dev, q2_dev], y_dev),
                   batch_size=4, nb_epoch=20, shuffle=True, )

    # compute final accuracy on training and test sets
    scores = net.evaluate([q1_test, q2_test], y_test)
    print("\n%s: %.2f%%" % (net.metrics_names[2], scores[2] * 100))

    # print(hist.history['acc'])
    # print(hist.history['val_acc'])
